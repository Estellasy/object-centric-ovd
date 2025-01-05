# 准备开放词汇目标检测的数据集

我们在[COCO](https://cocodataset.org/)和[LVIS v1.0](https://www.lvisdataset.org/)数据集上进行开放词汇检测(OVD)实验。
我们使用[ImageNet-21K](https://www.image-net.org/download.php)中与LVIS类别重叠的997个类别的子集，以及[COCO captions](https://cocodataset.org/)
数据集分别用于LVIS和COCO实验中的图像级监督(ILS)。我们使用广义零样本检测设置，其中分类器同时包含基础类别和新类别。
在开始处理之前，请从官方网站下载所需的数据集，并将它们放置或软链接到`$object-centric-ovd/datasets/`目录下。

```
object-centric-ovd/datasets/
    lvis/
    coco/
    imagenet/
    zeroshot_weights/
```
`zeroshot_weights/`包含在代码库中，其中包含开放词汇零样本分类器头部的预处理权重。有关如何准备这些权重的详细信息，请参见下面的零样本权重部分。

下载COCO图像、COCO和LVIS标注文件，并按如下方式放置：

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
```

## COCO开放词汇

OVD训练所需的标注文件`instances_train2017_seen_2_oriorder.json`、`instances_train2017_seen_2_oriorder_cat_info.json`
和评估文件`instances_val2017_all_2_oriorder.json`，以及图像级监督的标注文件`captions_train2017_tags_allcaps_pis.json`
可以从[这里](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EhJUv1cVKJtCnZJzsvGgVwYBgxP6M9TWsD-PBb_KgOjhmQ?e=iYkfDZ)下载。

用于基于区域知识蒸馏(RKD)的类别无关MAVL建议框的CLIP图像特征，以及用于伪图像级监督(PIS)的特定类别建议框可以从[这里](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EeJuo844j8FIsnuiX3wBxCgBcBR2MSjbhiLCuA4OC2cSWg?e=5BeESO)下载。
解压文件`coco_props.tar.gz`并按如下所示放置在相应位置：

```
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
        captions_train2017_tags_allcaps_pis.json
    zero-shot/
        instances_train2017_seen_2_oriorder.json
        instances_val2017_all_2_oriorder.json
        instances_train2017_seen_2_oriorder_cat_info.json
MAVL_proposals
    coco_props/
        classagnostic_distilfeats/
            000000581921.pkl
            000000581929.pkl
            ....
        class_specific/
            000000581921.pkl
            000000581929.pkl
            ....
```

或者，按照以下说明从COCO标准标注生成它们。我们遵循[Detic](https://github.com/facebookresearch/Detic)的代码库进行数据集准备。

1) COCO标注

按照[OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb)的工作，我们首先创建开放词汇COCO分割。
转换后的文件应按如下方式放置：
```
coco/
    zero-shot/
        instances_train2017_seen_2.json
        instances_val2017_all_2.json
```
然后使用以下命令对这些标注进行预处理，以便于评估：

```
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
```

2) 从COCO-Captions获取图像级标签的标注

最后需要准备的标注是`captions_train2017_tags_allcaps_pis.json`。
对于图像级监督，我们使用COCO-captions标注并用MAVL预测进行过滤。
首先使用以下命令从COCO-captions生成图像级标签：
```
python tools/get_cc_tags.py --cc_ann datasets/coco/annotations/captions_train2017.json 
    --out_path datasets/coco/annotations/captions_train2017_tags_allcaps.json  
    --allcaps --convert_caption --cat_path datasets/coco/annotations/instances_val2017.json
```
这将创建`datasets/coco/captions_train2017_tags_allcaps.json`。

要忽略COCO中不包含在已见(65)+未见(17)类别中的其余类别，使用标志`IGNORE_ZERO_CATS`的`instances_train2017_seen_2_oriorder_cat_info.json`。
可以通过以下命令创建：
```
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_oriorder.json
```

3) PIS的建议框

使用COCO-captions的图像级标签`captions_train2017_tags_allcaps_pis.json`，
使用以下命令通过MAVL生成PIS的伪建议框。从外部子模块下载检查点。
```
python tools/get_ils_labels.py -ckpt <mavl预训练权重路径> -dataset coco -dataset_dir datasets/coco
        -output datasets/MAVL_proposals/coco_props/class_specific
```
特定类别的伪建议框将作为单独的pickle文件存储在每个图像中。

使用以下命令，用MAVL的伪建议框过滤COCO-captions的图像级标注`captions_train2017_tags_allcaps.json`：
```
python tools/update_cc_pis_annotations.py --ils_path datasets/MAVL_proposals/coco_props/class_specific 
    --cc_ann_path datasets/coco/annotations/captions_train2017_tags_allcaps.json 
    --output_ann_path datasets/coco/annotations/captions_train2017_tags_allcaps_pis.json
```
这将创建`datasets/coco/captions_train2017_tags_allcaps_pis.json`。

4) 用于RKD的区域和CLIP嵌入

为训练集中的图像生成类别无关的建议框。类别无关的伪建议框和相应的CLIP图像特征将作为单独的pickle文件存储在每个图像中。
注意：这不依赖于步骤2和3。
```
python tools/get_rkd_clip_feat.py -ckpt <mavl预训练权重路径> -dataset coco -dataset_dir datasets/coco
        -output datasets/MAVL_proposals/coco_props/classagnostic_distilfeats
```

## LVIS开放词汇

OVD训练的标注文件`lvis_v1_train_norare`可以从[这里](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EjaR4-EQFmVNhaYmsJhbwKMBciHdSFd8Z2J7byTplHAURA?e=tbzhUM)下载，
图像级监督的标注文件`imagenet_lvis_v1_pis`可以从[这里](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/Eg22Nzk-QelGgjhSKwtQ25QB_FgDR92VOQU3lr79uMXqqQ?e=FSsFnt)下载。
注意：提供的ImageNet标注`imagenet_lvis_image_info_pis`包含相应LVIS类别的MAVL类别特定预测，以加快训练速度。

用于ImageNet上基于区域知识蒸馏(RKD)的类别无关MAVL建议框的CLIP图像特征可以从[这里](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/El9e1sKtdgBHlSxs0rEul5IBi2gcQBGthxXo0u4u-PlNcQ?e=VbqHDY)下载。
LVIS图像上的MAVL建议框与为COCO生成的建议框相同，可以从[这里](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EeJuo844j8FIsnuiX3wBxCgBcBR2MSjbhiLCuA4OC2cSWg?e=5BeESO)下载。
解压文件`imagenet_distil_feats.tar`和`coco_props.tar.gz`，并按如下所示放置在相应位置：

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
    lvis_v1_train_norare.json
    lvis_v1_train_norare_cat_info.json
coco/
    train2017/
    val2017/
imagenet/
    ImageNet-LVIS/
        n13000891/
        n15075141/
    annotations
        imagenet_lvis_image_info_pis.json
MAVL_proposals
    lvis_props/
        classagnostic_distilfeats/
            coco_distil_feats/
                    000000581921.pkl
                    000000581929.pkl
                    ....
            imagenet_distil_feats/
                    n13000891/
                        n13000891_995.pkl
                        n13000891_999.pkl
                        ....
                    n15075141/
                        n15075141_9997.pkl
                        n15075141_999.pkl
                        ....
        class_specific/
            imagenet_lvis_props/
                    n13000891/
                        n13000891_995.pkl
                        n13000891_999.pkl
                        ....
                    n15075141/
                        n15075141_9997.pkl
                        n15075141_999.pkl
                        ....
```

通过解压Image-Net21k准备`ImageNet-LVIS`目录。注意我们使用winter-21版本。
```
python tools/unzip_imagenet_lvis.py --dst_path datasets/imagenet/ImageNet-LVIS
```

或者，按照以下说明从标准标注生成它们。
你可以使用以下方法准备开放词汇LVIS训练集：

1) LVIS标注
```
python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
```

这将生成`datasets/lvis/lvis_v1_train_norare.json`。

`lvis_v1_train_norare_cat_info.json`由联邦损失使用。可以通过以下命令创建：
```
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train_norare.json
```

2) PIS的标注

对于图像级监督的Image-Net标注，我们首先解压LVIS重叠类别并将它们转换为LVIS标注格式。

```
mkdir imagenet/annotations
python tools/create_imagenetlvis_json.py --imagenet_path datasets/imagenet/ImageNet-LVIS
 --out_path datasets/imagenet/annotations/imagenet_lvis_image_info.json
```
这将创建`datasets/imagenet/annotations/imagenet_lvis_image_info.json`。

3) PIS的建议框

使用ImageNet的图像级标签`imagenet_lvis_image_info.json`，
使用以下命令通过MAVL生成PIS的伪建议框。从外部子模块下载检查点。
```
python tools/get_ils_labels.py -ckpt <mavl预训练权重路径> -dataset imagenet_lvis
        -dataset_dir datasets/imagenet -output datasets/MAVL_proposals/lvis_props/class_specific/imagenet_lvis_props
```
特定类别的伪建议框将作为单独的pickle文件存储在每个图像中。

要生成带有MAVL类别特定预测的标注，运行以下命令。此命令从图像级pkl文件创建单个json文件，用于ILS。
这不是强制性的，但这样做可以提高训练效率（而不是在数据加载器中从单独的pickle文件加载伪建议框）。
如果使用单个json文件，请不要在配置中设置`PIS_PROP_PATH`路径。
```
python tools/create_lvis_ils_json.py -dataset_dir datasets/imagenet
    -prop_path datasets/MAVL_proposals/lvis_props/class_specific/imagenet_lvis_props
    -target_path datasets/imagenet/annotations/imagenet_lvis_image_info_pis.json
```

4) 用于RKD的区域和CLIP嵌入

为训练集中的图像生成类别无关的建议框。类别无关的伪建议框和相应的CLIP图像特征将作为单独的pickle文件存储在每个图像中。
注意：这不依赖于步骤2和3。

对于LVIS，使用从COCO生成的相同特征，因为图像相同且预测是类别无关的
```
ln -s datasets/MAVL_proposals/coco_props/classagnostic_distilfeats
        datasets/MAVL_proposals/lvis_props/classagnostic_distilfeats/coco_distil_feats
```
对于ImageNet-LVIS，
```
python tools/get_rkd_clip_feat.py -ckpt <mavl预训练权重路径> -dataset imagenet_lvis
        -dataset_dir datasets/imagenet 
        -output datasets/MAVL_proposals/lvis_props/classagnostic_distilfeats/imagenet_distil_feats
```

## 零样本权重

我们使用查询`a photo of a {category}`来计算分类器的测试嵌入。
```
zeroshot_weights/
    coco_clip_a+photo+cname.npy
    lvis_v1_clip_a+photo+cname.npy
```
COCO的权重可以通过以下命令生成：
```
python tools/dump_clip_features.py --ann datasets/coco/annotations/instances_val2017.json --out_path zeroshot_weights/coco_clip_a+photo+cname.npy --prompt photo
```
对于LVIS，命令是：
```
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path zeroshot_weights/lvis_v1_clip_a+photo+cname.npy --prompt photo
```