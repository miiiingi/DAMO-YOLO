import random

import torch
import torch.nn as nn


class ONNX_NMS(torch.autograd.Function):
    """ONNX Standard NonMaxSuppression (TensorRT 호환)"""

    @staticmethod
    def forward(
        ctx, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    ):
        """
        Args:
            boxes: [batch, num_boxes, 4] - xyxy format
            scores: [batch, num_classes, num_boxes]
            max_output_boxes_per_class: tensor([value])
            iou_threshold: tensor([value])
            score_threshold: tensor([value])
        """
        # Forward는 ONNX export용 더미 출력
        device = boxes.device
        batch = scores.shape[0]
        num_det = torch.randint(0, 100, (1,)).item()

        # 더미 결과 생성
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)

        selected_indices = torch.cat(
            [batches[None], zeros[None], idxs[None]], 0
        ).T.contiguous()

        return selected_indices.to(torch.int64)

    @staticmethod
    def symbolic(
        g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    ):
        """ONNX Standard NonMaxSuppression operator"""
        return g.op(
            "NonMaxSuppression",  # ★ ONNX 표준 연산자
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            center_point_box_i=0,  # 0: xyxy, 1: center+wh
        )


class ONNX_Standard_End2End(nn.Module):
    """ONNX 표준 NMS를 사용하는 End2End 모듈"""

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj], dtype=torch.int64).to(device)
        self.iou_threshold = torch.tensor([iou_thres], dtype=torch.float32).to(device)
        self.score_threshold = torch.tensor([score_thres], dtype=torch.float32).to(
            device
        )

        # xyxy 변환 매트릭스
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=self.device,
        )

    def forward(self, score, box):
        """
        Args:
            score: [batch, num_boxes, num_classes]
            box: [batch, num_boxes, 4]

        Returns:
            num_det: [batch, 1] INT32
            det_boxes: [batch, max_obj, 4] FLOAT
            det_scores: [batch, max_obj] FLOAT
            det_classes: [batch, max_obj] INT32
        """
        batch, num_boxes, num_classes = score.shape

        # Box 좌표 변환 (center -> xyxy)
        nms_box = box @ self.convert_matrix

        # Score 변환: [batch, num_boxes, num_classes] -> [batch, num_classes, num_boxes]
        nms_score = score.transpose(1, 2).contiguous()

        # ONNX NMS 적용
        selected_indices = ONNX_NMS.apply(
            nms_box, nms_score, self.max_obj, self.iou_threshold, self.score_threshold
        )

        # selected_indices: [num_selected, 3]
        # 각 row는 [batch_idx, class_idx, box_idx]

        if selected_indices.numel() == 0:
            # 검출 결과 없음
            num_det = torch.zeros((batch, 1), dtype=torch.int32, device=self.device)
            det_boxes = torch.zeros((batch, self.max_obj.item(), 4), device=self.device)
            det_scores = torch.zeros((batch, self.max_obj.item()), device=self.device)
            det_classes = torch.zeros(
                (batch, self.max_obj.item()), dtype=torch.int32, device=self.device
            )
            return num_det, det_boxes, det_scores, det_classes

        batch_inds, cls_inds, box_inds = selected_indices.unbind(1)

        # 선택된 boxes와 scores 추출
        selected_score = nms_score[batch_inds, cls_inds, box_inds].unsqueeze(1)
        selected_box = nms_box[batch_inds, box_inds, ...]

        # [box, score] 결합
        dets = torch.cat([selected_box, selected_score], dim=1)

        # Batch별로 정리
        batched_dets = dets.unsqueeze(0).repeat(batch, 1, 1)
        batch_template = torch.arange(
            0, batch, dtype=batch_inds.dtype, device=batch_inds.device
        )
        batched_dets = batched_dets.where(
            (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
            batched_dets.new_zeros(1),
        )

        batched_labels = cls_inds.unsqueeze(0).repeat(batch, 1)
        batched_labels = batched_labels.where(
            (batch_inds == batch_template.unsqueeze(1)), batched_labels.new_ones(1) * -1
        )

        # Padding 추가
        N = batched_dets.shape[0]
        batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
        batched_labels = torch.cat(
            (batched_labels, -batched_labels.new_ones((N, 1))), 1
        )

        # Score로 정렬
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)

        topk_batch_inds = torch.arange(
            batch, dtype=topk_inds.dtype, device=topk_inds.device
        ).view(-1, 1)
        batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
        det_classes = batched_labels[topk_batch_inds, topk_inds, ...]

        # 최종 출력 분리
        det_boxes, det_scores = batched_dets.split((4, 1), -1)
        det_scores = det_scores.squeeze(-1)

        # 유효한 검출 개수
        num_det = (det_scores > 0).sum(1, keepdim=True).int()

        return num_det, det_boxes, det_scores, det_classes.int()


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(
        self,
        model,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        device=None,
        ort=False,
        trt_version=10,
        with_preprocess=False,
    ):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.with_preprocess = with_preprocess
        self.model = model.to(device)

        self.end2end = ONNX_Standard_End2End(max_obj, iou_thres, score_thres, device)
        self.end2end.eval()

    def forward(self, x):
        if self.with_preprocess:
            x = x[:, [2, 1, 0], ...]  # BGR to RGB
            x = x * (1 / 255)  # Normalize to [0, 1]

        # 모델 추론 (실제 detection head 실행)
        x = self.model(x)
        # NMS 후처리
        x = self.end2end(x[0], x[1])
        return x
