#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import argparse
import sys
import os

import onnx
import torch
from loguru import logger
from torch import nn

from damo.base_models.core.end2end import End2End
from damo.base_models.core.ops import RepConv, SiLU
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils.model_utils import get_model_info, replace_module
from damo.base_models.core.end2end import TRT8_NMS

from Calibrator import MyCalibrator


def verify_onnx_model(onnx_path):
    """ONNX 모델이 제대로 생성되었는지 검증"""
    import onnx
    from onnx import checker

    try:
        model = onnx.load(onnx_path)
        checker.check_model(model)

        # 그래프 노드 분석
        graph = model.graph
        logger.info(f"Total nodes: {len(graph.node)}")

        real_ops = [n for n in graph.node if n.op_type.startswith("TRT::")]

        if len(real_ops) == 0:
            logger.error("TensorRT plugin node가 없습니다.")

        if len(real_ops) == 0:
            logger.warning("⚠️ 경고: 실제 연산 노드가 없습니다!")
            logger.warning("모델이 더미 출력만 생성하도록 export되었을 수 있습니다.")
            return False

        # Random 노드 확인
        random_ops = [n for n in graph.node if "Random" in n.op_type]
        if random_ops:
            logger.warning(f"⚠️ Random 노드 발견: {len(random_ops)}개")
            for node in random_ops[:3]:  # 처음 3개만 출력
                logger.warning(f"  - {node.name}: {node.op_type}")

        return len(real_ops) > 0

    except Exception as e:
        logger.error(f"ONNX 검증 실패: {e}")
        return False


@logger.catch
def build_trt_engine(
    onnx_file_path,
    dummy_data,
    max_workspace_size=1 << 28,
    fp16_mode=False,
    int8_mode=False,
    strip_weights=False,
    use_ensemble: bool = False,
):
    import tensorrt as trt

    """
    ONNX 모델 파일을 TensorRT 엔진으로 변환하고 저장합니다.
    :param onnx_file_path: ONNX 모델 파일 경로
    :param engine_file_path: 저장할 TensorRT 엔진 파일 경로
    :param max_workspace_size: 엔진 빌드 시 사용할 최대 워크스페이스 크기 (예: 256MB)
    :param fp16_mode: FP16 모드 사용 여부 (사용 가능한 하드웨어인 경우 성능 향상)
    """
    engine_path = onnx_file_path.replace(".onnx", ".trt")
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # ★ 프로파일링 강화

    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    try:
        with open(onnx_file_path, "rb") as model_file:
            onnx_model = model_file.read()
            logger.info("54")
            logger.info("Loading ONNX MODEL Successful!")
        parse_success = parser.parse(onnx_model)
        if not parse_success:
            print("ONNX 모델 파싱 중 오류가 발생했습니다:")
            for error_idx in range(parser.num_errors):
                print(parser.get_error(error_idx))
            raise RuntimeError("ONNX 모델을 파싱할 수 없습니다.")
    except:
        raise RuntimeError("ONNX 모델을 로딩할 수 없습니다.")

    # SPARSE_WEIGHTS (Ampere+), OBEY_PRECISION_CONSTRAINTS 플래그
    # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for input in inputs:
        logger.info(f"Model {input.name} shape: {input.shape} {input.dtype}")
    for output in outputs:
        logger.info(f"Model {output.name} shape: {output.shape} {output.dtype}")

    profile = builder.create_optimization_profile()
    input_name = inputs[0].name  # 보통 "input" 이라고 지정됨
    min_shape = (1, dummy_data.shape[1], dummy_data.shape[2], dummy_data.shape[3])
    opt_shape = tuple(dummy_data.shape)
    max_shape = tuple(dummy_data.shape)

    logger.info(
        f"Optimization Profile for '{input_name}': min={min_shape}, opt={opt_shape}, max={max_shape}"
    )
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 빌더 설정: workspace size 및 FP16 모드 활성화 (하드웨어 지원 시)
    # builder.max_workspace_size = max_workspace_size
    float16 = fp16_mode
    int8 = int8_mode
    logger.info(f"float16: {fp16_mode}, int8: {int8_mode}")
    if float16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif int8:
        config.set_flag(trt.BuilderFlag.INT8)

        # INT8 모드에서는 캘리브레이터를 반드시 등록해야 합니다.
        # calibration_data는 미리 준비한 numpy 배열 리스트 또는 배열을 전달합니다.
        input_shape = dummy_data.shape  # 첫 번째 입력 텐서의 shape 사용
        calibrator = MyCalibrator(dummy_data.cpu().numpy(), 1, input_shape)
        config.int8_calibrator = calibrator

    if strip_weights:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)

    # TensorRT 엔진 빌드 (엔진 빌드 시 모델 최적화 수행)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT 엔진 빌드에 실패했습니다.")

    # 생성된 엔진을 파일로 직렬화 및 저장
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    logger.info(f"TensorRT 엔진이 성공적으로 저장되었습니다: {engine_path}")
    return engine_path


def export_to_onnx(model, dummy_input, onnx_file_path, opset_version, args):
    """ONNX export with custom ops"""

    model.eval()
    logger.info("=== 모델 추론 테스트 ===")

    with torch.no_grad():
        test_output = model(dummy_input)
        logger.info(f"✅ 모델 추론 성공! 출력 개수: {len(test_output)}")

        for idx, out in enumerate(test_output):
            if isinstance(out, torch.Tensor):
                logger.info(f"  Output {idx}: shape={out.shape}, dtype={out.dtype}")

    logger.info("=== ONNX Export 시작 ===")

    # ★ Custom opset 등록 (TensorRT plugin을 위해 필수)
    torch.onnx.register_custom_op_symbolic(
        "TRT::INMSLayer", TRT8_NMS.symbolic, opset_version
    )

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=opset_version,
        input_names=[args.input],
        output_names=(
            ["num_dets", "det_boxes", "det_scores", "det_classes"]
            if args.end2end
            else [args.output]
        ),
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        export_params=True,
        # ★ custom_opsets 추가
        custom_opsets={"TRT": 1} if args.end2end else None,
        # ★ operator_export_type 설정
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        dynamo=False,
    )

    logger.info(f"✅ ONNX 모델 저장: {onnx_file_path}")
    exit(0)


def make_parser():
    parser = argparse.ArgumentParser('damo converter deployment toolbox')
    # mode part
    parser.add_argument('--mode',
                        default='onnx',
                        type=str,
                        help='onnx, trt_16 or trt_32')
    # model part
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='expriment description file',
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='if true, export without postprocess'
    )
    parser.add_argument('-c',
                        '--ckpt',
                        default=None,
                        type=str,
                        help='ckpt path')
    parser.add_argument('--trt',
                        action='store_true',
                        help='whether convert onnx into tensorrt')
    parser.add_argument(
        '--trt_type', type=str, default='fp32',
        help='one type of int8, fp16, fp32')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='inference image batch nums')
    parser.add_argument('--img_size',
                        type=int,
                        default='640',
                        help='inference image shape')
    # onnx part
    parser.add_argument('--input',
                        default='images',
                        type=str,
                        help='input node name of onnx model')
    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output node name of onnx model')
    parser.add_argument('-o',
                        '--opset',
                        default=11,
                        type=int,
                        help='onnx opset version')
    parser.add_argument(
        "--fp16", action="store_true", help="using float 16 optimization"
    )
    parser.add_argument("--int8", action="store_true", help="using int 8 optimization")
    parser.add_argument('--end2end',
                        action='store_true',
                        help='export end2end onnx')
    parser.add_argument('--ort',
                        action='store_true',
                        help='export onnx for onnxruntime')
    parser.add_argument('--trt_eval',
                        action='store_true',
                        help='trt evaluation')
    parser.add_argument('--with-preprocess',
                        action='store_true',
                        help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all',
                        type=int,
                        default=100,
                        help='topk objects for every images')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='iou threshold for NMS')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.05,
                        help='conf threshold for NMS')
    parser.add_argument('--device',
                        default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def main():
    args = make_parser().parse_args()

    logger.info('args value: {}'.format(args))
    onnx_name = args.config_file.split('/')[-1].replace('.py', '.onnx')

    if args.end2end:
        onnx_name = onnx_name.replace('.onnx', '_end2end.onnx')

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (
        device.type == 'cpu' and args.trt_type != 'fp32'
    ), '{args.trt_type} only compatible with GPU export, i.e. use --device 0'
    # init and load model
    config = parse_config(args.config_file)
    config.merge(args.opts)
    if args.benchmark:
        config.model.head.export_with_post = False

    if args.batch_size is not None:
        config.test.batch_size = args.batch_size

    # build model
    model = build_local_model(config, device)
    # load model paramerters
    ckpt = torch.load(args.ckpt, map_location=device)

    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=True)
    logger.info(f'loading checkpoint from {args.ckpt}.')

    model = replace_module(model, nn.SiLU, SiLU)

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    info = get_model_info(model, (args.img_size, args.img_size))
    logger.info(info)
    # decouple postprocess
    model.head.nms = False

    if args.end2end:
        trt_version = 10
        model = End2End(model,
                        max_obj=args.topk_all,
                        iou_thres=args.iou_thres,
                        score_thres=args.conf_thres,
                        device=device,
                        ort=args.ort,
                        trt_version=trt_version,
                        with_preprocess=args.with_preprocess)

    dummy_input = torch.randn(args.batch_size, 3, args.img_size,
                              args.img_size).to(device)
    if not os.path.isfile(onnx_name):
        export_to_onnx(
            model, dummy_input, onnx_name, opset_version=args.opset, args=args
        )
        logger.info("Complete Saving ONNX File...")
        logger.info("Loading ONNX File...")
        onnx_model = onnx.load(onnx_name)
        logger.info("Complete Loading ONNX File...")
        # Fix output shape
        if args.end2end and not args.ort:
            shapes = [
                args.batch_size,
                1,
                args.batch_size,
                args.topk_all,
                4,
                args.batch_size,
                args.topk_all,
                args.batch_size,
                args.topk_all,
            ]
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))
        # try:
        #     import onnxsim

        #     logger.info("Starting to simplify ONNX...")
        #     # ★ onnxsim에 추가 옵션 전달
        #     onnx_model, check = onnxsim.simplify(
        #         onnx_model,
        #     )
        #     assert check, "check failed"
        # except Exception as e:
        #     logger.info(f"simplify failed: {e}")
        # logger.info("Saving SIM ONNX File...")
        # onnx.save(onnx_model, onnx_name)
        # logger.info("Complete Saving SIM ONNX File...")
        # logger.info(f"onnx name: {onnx_name}")

    # ★ 검증 추가
    if not verify_onnx_model(onnx_name):
        logger.error("ONNX 모델이 올바르지 않습니다. export 설정을 확인하세요.")
        raise RuntimeError("Invalid ONNX model exported")

    if args.trt:
        trt_name = build_trt_engine(
            onnx_name, dummy_input, fp16_mode=args.fp16, int8_mode=args.int8
        )
        if args.trt_eval:
            from trt_eval import trt_inference
            logger.info('start trt inference on coco validataion dataset')
            trt_inference(config, trt_name, args.img_size, args.batch_size,
                          args.conf_thres, args.iou_thres, args.end2end)


if __name__ == '__main__':
    main()
