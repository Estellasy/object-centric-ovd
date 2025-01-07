import os
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .lvis_v1 import custom_register_lvis_instances

categories_seen = [
    {'id': 1, 'name': 'crazing'},
    {'id': 2, 'name': 'inclusion'},
    {'id': 3, 'name': 'patches'},
    {'id': 4, 'name': 'pitted_surface'},
    {'id': 5, 'name': 'rolled-in_scale'},
    {'id': 6, 'name': 'scratches'}
]

categories_unseen = [
    {'id': 6, 'name': 'scratches'}
]


def _get_metadata(cat):
    if cat == 'all':
        return _get_builtin_metadata('coco')
    elif cat == 'seen':
        id_to_name = {x['id']: x['name'] for x in categories_seen}
    else:
        assert cat == 'unseen'
        id_to_name = {x['id']: x['name'] for x in categories_unseen}

    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_NEUDET = {
    "neudet_zeroshot_train": ("neudet_coco/train2017", "neudet_coco/zero-shot/instances_train2017_seen_2.json", 'seen'),
    "neudet_zeroshot_val": ("neudet_coco/val2017", "neudet_coco/zero-shot/instances_val2017_unseen_2.json", 'unseen'),
    "neudet_not_zeroshot_val": ("neudet_coco/val2017", "neudet_coco/zero-shot/instances_val2017_seen_2.json", 'seen'),
    "neudet_generalized_zeroshot_val": ("neudet_coco/val2017", "neudet_coco/zero-shot/instances_val2017_all_2.json", 'all'),
    "neudet_zeroshot_train_oriorder": (
        "neudet_coco/train2017", "neudet_coco/zero-shot/instances_train2017_all_2.json", 'all'),
    "neudet_test": ("neudet_coco/val2017", "neudet_coco/annotations/instances_val2017.json", 'all'),
    "neudet_train": ("neudet_coco/train2017", "neudet_coco/annotations/instances_train2017.json", 'all'),
}

for key, (image_root, json_file, cat) in _PREDEFINED_SPLITS_NEUDET.items():
    register_coco_instances(
        key,
        _get_metadata(cat),
        os.path.join("data", json_file) if "://" not in json_file else json_file,
        os.path.join("data", image_root),
    )


"""处理额外的caption数据集
_CUSTOM_SPLITS_NEUDET = {
    "neudet_caption_train_tags": ("neudet/train2017/", "neudet/annotations/captions_train2017_tags_allcaps_pis.json"),
    "neudet_caption_val_tags": ("neudet/val2017/", "neudet/annotations/captions_val2017_tags_allcaps.json"), }

for key, (image_root, json_file) in _CUSTOM_SPLITS_NEUDET.items():
    custom_register_lvis_instances(
        key,
        _get_builtin_metadata('coco'),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )   
"""
