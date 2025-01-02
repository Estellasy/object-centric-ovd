from typing import Dict, List, Optional, Tuple
import torch
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import Instances, Boxes
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from torch.cuda.amp import autocast


@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    """
    Add image labels
    """

    @configurable
    def __init__(
            self,
            fp16=False,     # 是否使用半精度训练
            roi_head_name='',   # ROI Head名称，这里用的是CustomRes5ROIHeads
            distillation=False, # 是否使用知识蒸馏
            **kwargs):
        """
        """
        self.roi_head_name = roi_head_name
        self.return_proposal = False    # 为什么这里是False
        self.fp16 = fp16
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        self.distillation = distillation

    @classmethod
    def from_config(cls, cfg):  # 从cfg中添加自定义配置
        ret = super().from_config(cfg)
        ret.update({
            'fp16': cfg.FP16,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'distillation': cfg.MODEL.DISTILLATION,
        })
        return ret

    def inference(  # 推理过程
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)  # 预处理图像 这里是Detectron的方法
        features = self.backbone(images.tensor) # 提取特征
        proposals, _ = self.proposal_generator(images, features, None)  # 生成候选区域（proposal_generator详细代码）
        results, _ = self.roi_heads(images, (features, None), proposals)    # ROI头处理
        if do_postprocess:  # 后处理
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] # 获取gt实例

        if self.fp16:
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        proposals, proposal_losses = self.proposal_generator(   # 调用proposal_generator生成候选区域
            images, features, gt_instances)

        if self.distillation:   # 知识蒸馏特征处理 提取clip特征（疑问：这里是不是提取gt的clip特征）
            distill_clip_features = self.get_clip_image_features(batched_inputs, images)
        else:
            distill_clip_features = None

        proposals, detector_losses = self.roi_heads(    # 这一部分是如何优化的
            images, (features, distill_clip_features), proposals, gt_instances, ann_type=ann_type)

        if self.vis_period > 0: # 可视化
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.return_proposal:    # True：同时返回提议框（proposals）和损失字典
            return proposals, losses
        else:                       # 只返回损失字典 在标准训练中，我们只需要损失来更新模型 提议框主要用于调试或可视化
            return losses

    def get_clip_image_features(self, batched_inputs, images):
        # 收集每张图片的CLIP特征
        # 使用 GT 区域的 CLIP 特征作为"教师信号" 实现知识从 CLIP 到检测器的迁移
        image_features = []
        region_boxes = []
        for n, image_size in enumerate(images.image_sizes):
            # 获取预计算的CLIP特征和对应区域（这里获取的是gt box的特征还是什么？不太理解）
            image_features.append(batched_inputs[n]['distill_feats'][1].to(images[n].device))       # [1]: 预计算的 CLIP 特征
            region_boxes.append(Boxes(batched_inputs[n]['distill_feats'][0].to(images[n].device)))  # [0]: GT 边界框坐标
        image_features = torch.cat(image_features, 0)
        return region_boxes, image_features
