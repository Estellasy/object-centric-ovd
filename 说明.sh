# 准备零样本分类器权重
python tools/dump_clip_features.py \
    --ann /home/siyi/project/object-centric-ovd/datasets/neudet/annotations/all.json \
    --out_path datasets/zeroshot_weights/neudet_all_clip_a+photo+cname.npy \
    --prompt photo

# 生成OVD所需的特殊标注：
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/neudet/annotations/all.json --cat_path datasets/neudet/annotations/all.json


进度说明：
- 01-06 完成数据集准备，生成CLIP特征
- 完成零样本分类器权重准备
- 下一步：模型训练和知识蒸馏过程demo编写

- 模型训练和知识蒸馏过程


- 理清数据集准备过程（准备好文件后，如何加载）
- 模型训练过程代码重写，移植已训练好的coco部分

模型训练：
- 首先跑通OVD RKD baseline

python train_net.py --num-gpus 1 \
--config-file configs/neudet/neudet_OVD_Base_RKD.yaml \
OUTPUT_DIR training_dir/ovd_r50

构造pkl文件