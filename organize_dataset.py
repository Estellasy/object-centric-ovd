import os
import json
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import tqdm

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def split_coco_dataset(all_json_path, train_ratio=0.8, random_seed=42):
    """
    将数据集划分为训练集和验证集
    
    Args:
        all_json_path: 原始json文件路径
        train_ratio: 训练集比例
        random_seed: 随机种子
    """
    # 加载原始数据
    data = load_json(all_json_path)
    
    # 准备新的数据结构
    train_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data.get('categories', []),
        'images': [],
        'annotations': []
    }
    val_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data.get('categories', []),
        'images': [],
        'annotations': []
    }

    # 获取所有图像ID
    image_ids = [img['id'] for img in data['images']]
    
    # 划分训练集和验证集
    train_ids, val_ids = train_test_split(
        image_ids, 
        train_size=train_ratio, 
        random_state=random_seed
    )
    
    # 转换为集合以提高查找效率
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    # 分配图像
    for img in data['images']:
        if img['id'] in train_ids_set:
            train_data['images'].append(img)
        else:
            val_data['images'].append(img)

    # 分配标注
    for ann in data['annotations']:
        if ann['image_id'] in train_ids_set:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)

    return train_data, val_data

def create_coco_structure(source_dir, target_dir, all_json_path):
    """
    组织数据集为COCO格式
    
    Args:
        source_dir: 源数据目录
        target_dir: 目标目录
        all_json_path: 原始json文件路径
    """
    # 创建必要的目录
    coco_dir = Path(target_dir)
    train_dir = coco_dir / 'train2017'
    val_dir = coco_dir / 'val2017'
    anno_dir = coco_dir / 'annotations'

    for dir_path in [train_dir, val_dir, anno_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 划分数据集
    print("Splitting dataset...")
    train_data, val_data = split_coco_dataset(all_json_path)

    # 保存标注文件
    print("Saving annotations...")
    save_json(train_data, anno_dir / 'instances_train2017.json')
    save_json(val_data, anno_dir / 'instances_val2017.json')

    # 复制图像文件
    source_images = Path(source_dir)
    
    # 创建训练集和验证集图像ID的集合
    train_image_ids = {img['file_name'] for img in train_data['images']}
    val_image_ids = {img['file_name'] for img in val_data['images']}

    print("Copying training images...")
    for img_file in tqdm.tqdm(list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))):
        if img_file.name in train_image_ids:
            shutil.copy2(img_file, train_dir / img_file.name)
        elif img_file.name in val_image_ids:
            shutil.copy2(img_file, val_dir / img_file.name)

if __name__ == "__main__":
    source_directory = "data/neudet/images"
    target_directory = "data/neudet_coco"
    all_json_path = "data/neudet/annotations/all.json"
    
    create_coco_structure(source_directory, target_directory, all_json_path)
    print("Dataset organization completed!")