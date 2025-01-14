from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from .custom_fast_rcnn import CustomFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class CustomRes5ROIHeads(Res5ROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        # 计算输出通道数
        stage_channel_factor = 2 ** 3   # Res5层的通道扩散因子
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor

        # 配置参数
        self.with_image_labels = cfg.WITH_IMAGE_LABELS
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        # box预测
        self.box_predictor = CustomFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def forward(self, images, features, proposals, targets=None, ann_type='box'):
        del images

        features, distill_clip_features = features  # 分离普通特征和CLIP特征
        if self.training:
            if ann_type == 'box':   # 训练阶段处理
                proposals = self.label_and_sample_proposals(proposals, targets) # 使用边界框标注
            else:
                proposals = self.get_top_proposals(proposals)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform([features[f] for f in self.in_features], proposal_boxes)
        predictions = self.box_predictor(box_features.mean(dim=[2, 3])) # ROI特征提取

        # 知识蒸馏
        if self.training and distill_clip_features is not None:
            # distilling image embedding
            # 提取区域特征
            distil_regions, distill_clip_embeds = distill_clip_features
            region_level_features = self._shared_roi_transform([features[f] for f in self.in_features], distil_regions)
            # 计算图像嵌入
            image_embeds = region_level_features.mean(dim=[2, 3])
            # image distillation
            # 特征投影和归一化
            proj_image_embeds = self.box_predictor.cls_score.linear(image_embeds)
            norm_image_embeds = F.normalize(proj_image_embeds, p=2, dim=1)
            normalized_clip_embeds = F.normalize(distill_clip_embeds, p=2, dim=1)
            distill_features = (norm_image_embeds, normalized_clip_embeds)
        else:
            distill_features = None

        if self.training:
            del features
            if ann_type != 'box':
                image_labels = [x._pos_category_ids for x in targets]
                losses = self.box_predictor.image_label_losses(
                    predictions, proposals, distill_features, image_labels)
            else:
                losses = self.box_predictor.losses(
                    (predictions[0], predictions[1]), proposals, distill_features)
                if self.with_image_labels:
                    assert 'pms_loss' not in losses
                    losses['pms_loss'] = predictions[0].new_zeros([1])[0]
            return proposals, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        return proposals
