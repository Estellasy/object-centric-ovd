import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import json

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.save_proposal_boxes import SaveProposalBoxes as SaveRKDFeats

import clip

# 这里需要load adapter clip 模型

def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ckpt", "--adapter_path", default=None, required=False,
                    help="The path to adapter clip model weights. If not provided, will use original CLIP model.")
    ap.add_argument("-dataset", "--dataset_name", required=False, default='coco',
                    help="The dataset name to generate the ILS labels for. Supported datasets are "
                         "['coco', 'imagenet_lvis']")
    ap.add_argument("-dataset_dir", "--dataset_base_dir_path", required=False, default='data/neudet_coco',
                    help="The dataset base directory path.")
    ap.add_argument("-output", "--output_dir_path", required=False,
                    default='datasets/clip_proposals/neudet_props/clip_distilfeats',
                    help="Path to save the ILS labels.")

    args = vars(ap.parse_args())
    return args


def crop_region(image, box):
    left, top, right, bottom = box
    im_crop = image.crop((left, top, right, bottom))
    return im_crop



def parse_coco_annotations(dataset_anno):
    dataset = json.load(open(dataset_anno, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    
    id_to_filename = {
        img['id']: img['file_name'].split('.')[0]  # 去掉文件扩展名
        for img in dataset['images']
    }

    annos = {}
    for ann in dataset['annotations']:
        image_id = ann['image_id']
        image_name = id_to_filename[image_id]
        
        bbox = ann['bbox']
        x, y, w, h = bbox
        bbox_xyxy = [x, y, x + w, y + h]
        
        if image_name not in annos:
            annos[image_name] = []
            
        # 添加bbox信息
        annos[image_name].append({
            'bbox': bbox_xyxy,
            'category_id': ann['category_id']
        })

    return annos


def get_clip_features(image_path, boxes):
    # 获取box框
    curr_rkd_region_feats = []
    try:
        for box in boxes:
            im_crop = crop_region(image_path, box)
            cropped_region = clip_preprocessor(im_crop).unsqueeze(0).to("cpu")
            with torch.no_grad():
                image_features = clip_model.encode_image(cropped_region)
                clip_embeds = image_features.cpu()
                curr_rkd_region_feats.append((box, clip_embeds))
    except Exception as e:
        pass

    return curr_rkd_region_feats


def get_coco_rkd_clip_features(anno, dataset_dir, save_dir):
    train_images_path = f"{dataset_dir}/train2017"
    # The coco dataset must be setup correctly before running this script, see datasets/README.md for details
    assert os.path.exists(train_images_path)
    # Iterate over all the images, generate class-agnostic proposals and extract CLIP features
    dumper = SaveRKDFeats()
    rkd_region_feats = {}
    for i, image_name in enumerate(tqdm(os.listdir(train_images_path))):
        if i > 0 and i % 100 == 0:  # Save every 100 iterations
            dumper.update(rkd_region_feats)
            dumper.save(save_dir)
            rkd_region_feats = {}
        image_path = f"{train_images_path}/{image_name}"
        image_name_key = image_name.split('.')[0]
        # 获取CLIP特征
        boxes = anno[image_name_key]
        rkd_region_feats[image_name_key] = get_clip_features(image_path, boxes)
    dumper.update(rkd_region_feats)
    dumper.save(save_dir)
    print(f"Save CLIP features to {save_dir}")



# TODO 加载CLIP Adapter模型
"""
class CLIPAdapter(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # 先加载模型
        self.clip_model, _ = clip.load(params['clip_model'])
        self.clip_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)

        # Adapter
        if 'conv' in params['adapter']:
            adapter = Adapter(1024, c_type=params['adapter'], dtype=torch.half).cuda()
        elif params['adapter'] == 'fc':
            adapter = Adapter_FC(1024, dtype=torch.half).cuda()
        self.adapter = adapter

        self.text_encoder = self.clip_model.encode_text

    def load(self):
        self.clip_model, self.preprocess = clip.load(self.params['clip_model'])
        adapter_state = torch.load(self.params['adapter_path'])
        self.adapter.load_state_dict(adapter_state)

        self.clip_model.eval()
        self.adapter.eval()

        return self.clip_model, self.adapter, self.preprocess
"""


if __name__ == "__main__":
    args = parse_arguments()
    dataset_name = args["dataset_name"]
    dataset_base_dir = args["dataset_base_dir_path"]
    dataset_anno = f"{dataset_base_dir}/annotations/instances_train2017.json"
    output_dir = args["output_dir_path"]
    os.makedirs(output_dir, exist_ok=True)

    # Load Adapter CLIP model
    # TODO

    # Load CLIP model
    clip_model, clip_preprocessor = clip.load("RN50", device="cpu")
    annos = parse_coco_annotations(dataset_anno)
    # 生成RKD特征
    if dataset_name == "coco":
        get_coco_rkd_clip_features(annos, dataset_base_dir, output_dir)
    else:
        print(f"Only 'coco' and 'imagenet_lvis' datasets are supported.")
        raise NotImplementedError
    

"""
python tools/get_rkd_adapter_clip_feat.py -dataset coco \
    -dataset_dir data/neudet_coco \
    -output datasets/clip_proposals/neudet_props/clip_distilfeats
"""