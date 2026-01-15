# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils import get_model_info, vis, postprocess
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from damo.structures.bounding_box import BoxList
import time
from matplotlib import pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # ⭐ CUDA context 자동 생성

ctx = pycuda.autoinit.context


IMAGES=['png', 'jpg']
VIDEOS=['mp4', 'avi']


def is_image_file(filename):
    return filename.split(".")[-1].lower() in IMAGES


def allocate_buffers(
    engine: trt.ICudaEngine, context: trt.IExecutionContext, batch_size: int
):
    bindings = {}
    stream = cuda.Stream()

    output_tensors = {}
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        is_input = tensor_mode == trt.TensorIOMode.INPUT
        dims = engine.get_tensor_shape(tensor_name)
        # 요소 개수
        volume = 1
        for d in dims:
            volume *= d if d > 0 else 1
        volume *= batch_size
        np_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        torch_dtype = torch.from_numpy(np.array([], dtype=np_dtype)).dtype
        if is_input:
            host_mem = cuda.pagelocked_empty(volume, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings[tensor_name] = device_mem
        else:
            output_tensor = torch.empty(
                size=tuple(dims), dtype=torch_dtype, device="cuda"
            )
            output_tensors[tensor_name] = output_tensor
            bindings[tensor_name] = output_tensor.data_ptr()
    return bindings, stream, output_tensors


def do_inference(context: trt.IExecutionContext, stream: cuda.Stream) -> list:
    # for inp in inputs:
    #     cuda.memcpy_htod_async(inp["device"], inp["host"], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()


def infer_with_dynamic_batch(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    images: torch.Tensor,
    use_ensemble: bool = False,
) -> torch.Tensor:
    images = images.contiguous()
    N, C, H, W = images.shape
    input_name = engine.get_tensor_name(0)
    context.set_input_shape(input_name, (N, C, H, W))
    bindings, stream, output_tensors = allocate_buffers(
        engine, context, images.shape[0]
    )
    for binding in engine:
        context.set_tensor_address(binding, int(bindings[binding]))

    cuda.memcpy_dtod_async(
        dest=int(bindings["images"]),
        src=int(images.data_ptr()),
        size=images.nbytes,
        stream=stream,
    )

    do_inference(context, stream)

    return output_tensors


class Infer():

    def __init__(
        self,
        config,
        infer_size=[640, 640],
        device="cuda",
        output_dir="./",
        ckpt=None,
        end2end=False,
        defect_name=None,
    ):

        self.ckpt_path = ckpt
        suffix = ckpt.split('.')[-1]
        if suffix == 'onnx':
            self.engine_type = 'onnx'
        elif suffix == 'trt':
            self.engine_type = 'tensorRT'
        elif suffix in ['pt', 'pth']:
            self.engine_type = 'torch'
        self.end2end = end2end # only work with tensorRT engine
        self.output_dir = os.path.join(output_dir, defect_name)
        os.makedirs(self.output_dir, exist_ok=True)
        if torch.cuda.is_available() and device=='cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if "class_names" in config.dataset:
            self.class_names = config.dataset.class_names
        else:
            self.class_names = []
            for i in range(config.model.head.num_classes):
                self.class_names.append(str(i))
            self.class_names = tuple(self.class_names)

        self.infer_size = infer_size
        config.dataset.size_divisibility = 0
        self.config = config
        if self.engine_type == "tensorRT":
            context, engine = self._build_engine(self.config, self.engine_type)
            self.model = engine
            self.context = context
        else:
            self.model = self._build_engine(self.config, self.engine_type)

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert n == 1
        assert h<=target_size[0] and w<=target_size[1]
        target_size = [n, c, target_size[0], target_size[1]]
        pad_imgs = torch.zeros(*target_size)
        pad_imgs[:, :c, :h, :w].copy_(img)

        img_sizes = [img.shape[-2:]]
        pad_sizes = [pad_imgs.shape[-2:]]

        return ImageList(pad_imgs, img_sizes, pad_sizes)

    def _build_engine(self, config, engine_type):

        print(f'Inference with {engine_type} engine!')
        if engine_type == 'torch':
            model = build_local_model(config, self.device)
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt['model'], strict=True)
            for layer in model.modules():
                if isinstance(layer, RepConv):
                    layer.switch_to_deploy()
            model.eval()
        elif engine_type == 'tensorRT':
            context, engine = self.build_tensorRT_engine(self.ckpt_path)
            return context, engine
        elif engine_type == 'onnx':
            model, self.input_name, self.infer_size, _, _ = self.build_onnx_engine(self.ckpt_path)
        else:
            NotImplementedError(f'{engine_type} is not supported yet! Please use one of [onnx, torch, tensorRT]')

        return model

    def build_tensorRT_engine(self, trt_path):

        loggert = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(loggert)
        with open(trt_path, 'rb') as t:
            engine = runtime.deserialize_cuda_engine(t.read())
        context = engine.create_execution_context()
        return context, engine

    def build_onnx_engine(self, onnx_path):

        import onnxruntime

        session = onnxruntime.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        out_names = []
        out_shapes = []
        for idx in range(len(session.get_outputs())):
            out_names.append(session.get_outputs()[idx].name)
            out_shapes.append(session.get_outputs()[idx].shape)
        return session, input_name, input_shape[2:], out_names, out_shapes

    def preprocess(self, origin_img):

        img = transform_img(origin_img, 0,
                            **self.config.test.augment.transform,
                            infer_size=self.infer_size)
        # img is a image_list
        oh, ow, _  = origin_img.shape
        img = self._pad_image(img.tensors, self.infer_size)

        img = img.to(self.device)
        return img, (ow, oh)

    def postprocess(self, preds, image, origin_shape=None):

        if self.engine_type == 'torch':
            output = preds

        elif self.engine_type == 'onnx':
            scores = torch.Tensor(preds[0])
            bboxes = torch.Tensor(preds[1])
            output = postprocess(scores, bboxes,
                self.config.model.head.num_classes,
                self.config.model.head.nms_conf_thre,
                self.config.model.head.nms_iou_thre,
                image)
        elif self.engine_type == 'tensorRT':
            if self.end2end:
                nums = preds["num_dets"]
                boxes = preds["det_boxes"]
                scores = preds["det_scores"]
                pred_classes = preds["det_classes"]
                batch_size = boxes.shape[0]
                output = [None for _ in range(batch_size)]
                for i in range(batch_size):
                    img_h, img_w = image.image_sizes[i]
                    boxlist = BoxList(torch.Tensor(boxes[i][:nums[i][0]]),
                              (img_w, img_h),
                              mode='xyxy')
                    boxlist.add_field(
                        "objectness",
                        torch.Tensor(torch.ones_like(scores[i][: nums[i][0]])),
                    )
                    boxlist.add_field('scores', torch.Tensor(scores[i][:nums[i][0]]))
                    boxlist.add_field('labels',
                              torch.Tensor(pred_classes[i][:nums[i][0]] + 1))
                    output[i] = boxlist
            else:
                cls_scores = torch.Tensor(preds[0])
                bbox_preds = torch.Tensor(preds[1])
                output = postprocess(cls_scores, bbox_preds,
                             self.config.model.head.num_classes,
                             self.config.model.head.nms_conf_thre,
                             self.config.model.head.nms_iou_thre, image)

        output = output[0].resize(origin_shape)
        bboxes = output.bbox
        scores = output.get_field('scores')
        cls_inds = output.get_field('labels')

        return bboxes,  scores, cls_inds

    def forward(self, origin_image):

        image, origin_shape = self.preprocess(origin_image)
        with torch.no_grad():
            if self.engine_type == "torch":
                output = self.model(image)

            elif self.engine_type == "onnx":
                image_np = np.asarray(image.tensors.cpu())
                output = self.model.run(None, {self.input_name: image_np})

            elif self.engine_type == "tensorRT":
                output = infer_with_dynamic_batch(
                    self.model, self.context, image.tensors
                )

            bboxes, scores, cls_inds = self.postprocess(
                output, image, origin_shape=origin_shape
            )

        return bboxes, scores, cls_inds

    def visualize(self, image, bboxes, scores, cls_inds, conf, save_name='vis.jpg', save_result=True):
        vis_img = vis(image, bboxes, scores, cls_inds, conf, self.class_names)
        if save_result:
            save_path = os.path.join(self.output_dir, save_name)
            # print(f"save visualization results at {save_path}")
            cv2.imwrite(save_path, vis_img[:, :, ::-1])
        return vis_img


def make_parser():
    parser = argparse.ArgumentParser('DAMO-YOLO Demo')

    parser.add_argument('input_type',
                        default='image',
                        help="input type, support [image, video, camera]")
    parser.add_argument('-f',
                        '--config_file',
                        default=None,
                        type=str,
                        help='pls input your config file',)
    parser.add_argument('-p',
                        '--path',
                        default='./assets/dog.jpg',
                        type=str,
                        help='path to image or video')
    parser.add_argument('--camid',
                        type=int,
                        default=0,
                        help='camera id, necessary when input_type is camera')
    parser.add_argument('--engine',
                        default=None,
                        type=str,
                        help='engine for inference')
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help='device used to inference')
    parser.add_argument('--output_dir',
                        default='./demo',
                        type=str,
                        help='where to save inference results')
    parser.add_argument('--conf',
                        default=0.6,
                        type=float,
                        help='conf of visualization')
    parser.add_argument('--infer_size',
                        nargs='+',
                        type=int,
                        help='test img size')
    parser.add_argument('--end2end',
                        action='store_true',
                        help='trt engine with nms')
    parser.add_argument('--save_result',
                        default=True,
                        type=bool,
                        help='whether save visualization results')


    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    config = parse_config(args.config_file)
    input_type = args.input_type

    defect_name = args.path.split("/")[-1]
    infer_engine = Infer(
        config,
        infer_size=args.infer_size,
        device=args.device,
        output_dir=args.output_dir,
        ckpt=args.engine,
        end2end=args.end2end,
        defect_name=defect_name,
    )

    if input_type == 'image':

        if os.path.isdir(args.path):
            image_files = [
                os.path.join(args.path, f)
                for f in os.listdir(args.path)
                if is_image_file(f)
            ]
            image_files.sort()

            logger.info(f"Found {len(image_files)} images in folder")

            warmup_image_path = image_files[0]
            warmup_image = np.asarray(
                Image.open(warmup_image_path).convert("RGB")
            ).copy()
            # warm-up
            for _ in range(10):
                _, _, _ = infer_engine.forward(warmup_image)
            torch.cuda.synchronize()

            inf_time_list = []
            for img_path in image_files:
                logger.info(f"Inferencing {img_path}")
                torch.cuda.synchronize()
                start = time.perf_counter()
                origin_img = np.asarray(Image.open(img_path).convert("RGB")).copy()

                bboxes, scores, cls_inds = infer_engine.forward(origin_img)
                torch.cuda.synchronize()
                end = time.perf_counter()
                ms = round((end - start) * 1000)
                inf_time_list.append(ms)

                # infer_engine.visualize(
                #     origin_img,
                #     bboxes,
                #     scores,
                #     cls_inds,
                #     conf=args.conf,
                #     save_name=os.path.basename(img_path),
                #     save_result=True,
                # )

            print(
                f"min: {min(inf_time_list)}, max: {max(inf_time_list)}, avg: {sum(inf_time_list) / len(inf_time_list)}"
            )
            arr = np.array(inf_time_list)
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # -----------------------------
            # (1) Histogram + KDE + Mean/Median
            # -----------------------------
            axes[0].hist(arr, bins=30, density=True, alpha=0.6)

            x = np.linspace(arr.min(), arr.max(), 300)
            mean = arr.mean()
            std = arr.std()

            kde = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            axes[0].plot(x, kde)

            axes[0].axvline(mean, linestyle="--", label=f"Mean: {mean:.2f} ms")
            axes[0].axvline(
                np.median(arr), linestyle="-.", label=f"Median: {np.median(arr):.2f} ms"
            )

            axes[0].set_title("Inference Time Distribution")
            axes[0].set_xlabel("Inference time (ms)")
            axes[0].set_ylabel("Density")
            axes[0].legend()

            # -----------------------------
            # (2) Violin Plot
            # -----------------------------
            axes[1].violinplot(arr, vert=False, showmeans=True, showmedians=True)
            axes[1].set_title("Inference Time Violin Plot")
            axes[1].set_xlabel("Inference time (ms)")

            plt.tight_layout()
            ext = args.engine.split(".")[-1]
            save_name = args.engine.replace(ext, "png")
            plt.savefig(save_name)
            plt.show()

        else:
            origin_img = np.asarray(Image.open(args.path).convert("RGB")).copy()
            bboxes, scores, cls_inds = infer_engine.forward(origin_img)

            vis_res = infer_engine.visualize(
                origin_img,
                bboxes,
                scores,
                cls_inds,
                conf=args.conf,
                save_name=os.path.basename(args.path),
                save_result=args.save_result,
            )

            if not args.save_result:
                cv2.namedWindow("DAMO-YOLO", cv2.WINDOW_NORMAL)
                cv2.imshow("DAMO-YOLO", vis_res)
                cv2.waitKey(0)

    elif input_type == 'video' or input_type == 'camera':
        cap = cv2.VideoCapture(args.path if input_type == 'video' else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        if args.save_result:
            save_path = os.path.join(args.output_dir, os.path.basename(args.path))
            print(f'inference result will be saved at {save_path}')
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (int(width), int(height)))
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                bboxes, scores, cls_inds = infer_engine.forward(frame)
                result_frame = infer_engine.visualize(frame, bboxes, scores, cls_inds, conf=args.conf, save_result=False)
                if args.save_result:
                    vid_writer.write(result_frame)
                else:
                    cv2.namedWindow("DAMO-YOLO", cv2.WINDOW_NORMAL)
                    cv2.imshow("DAMO-YOLO", result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break


if __name__ == '__main__':
    main()
