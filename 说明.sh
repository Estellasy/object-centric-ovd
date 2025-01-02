# 准备零样本分类器权重
python tools/dump_clip_features.py \
    --ann /home/siyi/project/object-centric-ovd/datasets/neudet/annotations/all.json \
    --out_path datasets/zeroshot_weights/neudet_all_clip_a+photo+cname.npy \
    --prompt photo

# 生成OVD所需的特殊标注：
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/neudet/annotations/all.json --cat_path datasets/neudet/annotations/all.json
