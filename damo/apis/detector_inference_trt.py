# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import os

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import tensorrt as trt
from damo.dataset.datasets.evaluation import evaluate
from damo.structures.bounding_box import BoxList
from damo.utils import postprocess
from damo.utils.timer import Timer

import pycuda.driver as cuda
import pycuda.autoinit

ctx = pycuda.autoinit.context

COCO_CLASSES = []
for i in range(4):
    COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def compute_on_dataset(config,
                       context,
                       stream,
                       data_loader,
                       device,
                       timer=None,
                       end2end=False):

    results_dict = {}
    cpu_device = torch.device('cpu')
    bindings = {}
    inputs = []
    outputs = []
    
    engine = context.engine
    
    # TensorRT 10.x: 모든 텐서 이름 가져오기
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        dtype = engine.get_tensor_dtype(tensor_name)
        shape = engine.get_tensor_shape(tensor_name)
        
        # 동적 shape 처리
        if -1 in shape:
            # 배치 크기를 1로 가정하고 shape 설정
            shape = tuple(1 if s == -1 else s for s in shape)
            context.set_input_shape(tensor_name, shape)
        
        is_input = (tensor_mode == trt.TensorIOMode.INPUT)
        
        if is_input:
            batch_size = shape[0]
        
        # 메모리 할당 크기 계산
        size = np.dtype(trt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        
        # GPU 메모리 할당
        allocation = cuda.mem_alloc(size)
        
        binding = {
            'index': i,
            'name': tensor_name,
            'dtype': np.dtype(trt.nptype(dtype)),
            'shape': list(shape),
            'allocation': allocation,
            'size': size
        }
        
        bindings[tensor_name] = allocation
        
        if is_input:
            inputs.append(binding)
        else:
            outputs.append(binding)
    
    # TensorRT 10.x: set_tensor_address 사용
    for tensor_name, allocation in bindings.items():
        context.set_tensor_address(tensor_name, int(allocation))

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            
            images_np = images.tensors.numpy()
            input_batch = images_np.astype(np.float32)
            
            # 동적 배치 크기 설정 (필요한 경우)
            actual_batch_size = input_batch.shape[0]
            if actual_batch_size != batch_size:
                input_shape = list(input_batch.shape)
                context.set_input_shape(inputs[0]['name'], input_shape)
                # 출력 shape도 업데이트
                for output in outputs:
                    new_shape = context.get_tensor_shape(output['name'])
                    output['shape'] = list(new_shape)

            trt_out = []
            for output in outputs:
                trt_out.append(np.zeros(output['shape'], output['dtype']))

            def predict(batch):
                # 입력 데이터를 GPU로 복사
                cuda.memcpy_htod_async(
                    inputs[0]['allocation'], 
                    np.ascontiguousarray(batch), 
                    stream
                )
                
                # 추론 실행
                context.execute_async_v3(stream_handle=stream.handle)
                stream.synchronize()
                
                # 출력 데이터를 CPU로 복사
                for o in range(len(trt_out)):
                    cuda.memcpy_dtoh_async(
                        trt_out[o], 
                        outputs[o]['allocation'], 
                        stream
                    )
                
                return trt_out

            pred_out = predict(input_batch)
            
            # trt with nms
            if end2end:
                nums = pred_out[0]
                boxes = pred_out[1]
                scores = pred_out[2]
                pred_classes = pred_out[3]
                batch_size_actual = boxes.shape[0]
                output = [None for _ in range(batch_size_actual)]
                
                for i in range(batch_size_actual):
                    img_h, img_w = images.image_sizes[i]
                    boxlist = BoxList(
                        torch.Tensor(boxes[i][:nums[i][0]]),
                        (img_w, img_h),
                        mode='xyxy'
                    )
                    boxlist.add_field(
                        'objectness',
                        torch.Tensor(np.ones_like(scores[i][:nums[i][0]]))
                    )
                    boxlist.add_field(
                        'scores',
                        torch.Tensor(scores[i][:nums[i][0]])
                    )
                    boxlist.add_field(
                        'labels',
                        torch.Tensor(pred_classes[i][:nums[i][0]] + 1)
                    )
                    output[i] = boxlist
            else:
                cls_scores = torch.Tensor(pred_out[0])
                bbox_preds = torch.Tensor(pred_out[1])
                output = postprocess(
                    cls_scores, bbox_preds,
                    config.model.head.num_classes,
                    config.model.head.nms_conf_thre,
                    config.model.head.nms_iou_thre,
                    images
                )

            if timer:
                torch.cuda.synchronize()
                timer.toc()

            output = [o.to(cpu_device) if o is not None else o for o in output]
        
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    
    return results_dict


def inference(
    config,
    context,
    stream,
    data_loader,
    dataset_name,
    iou_types=('bbox', ),
    box_only=False,
    device='cuda',
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    end2end=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    dataset = data_loader.dataset
    logger.info('Start evaluation on {} dataset({} images).'.format(
        dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(config, context, stream, data_loader, device,
                                     inference_timer, end2end)
    # convert to a list
    image_ids = list(sorted(predictions.keys()))
    predictions = [predictions[i] for i in image_ids]

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, 'predictions.pth'))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
