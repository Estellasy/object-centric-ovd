# 面向对象的开放词汇检测 (NeurIPS 2022)
论文 "[弥合开放词汇检测中对象级和图像级表示之间的差距](https://arxiv.org/abs/2207.03482)" 的官方代码库。

[Hanoona Rasheed](https://scholar.google.com/citations?user=yhDdEuEAAAAJ&hl=en&authuser=1&oi=sra), [Muhammad Maaz](https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra), [Muhammad Uzair Khattak](https://scholar.google.com/citations?user=M6fFL4gAAAAJ&hl=en&authuser=1), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![网站](https://img.shields.io/badge/Project-Website-87CEEB)](https://hanoonar.github.io/object-centric-ovd)
[![论文](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2207.03482)
[![Colab演示](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19LBqQg0cS36rTLL_TaXZ7Ka9KJGkxiSe?usp=sharing)
[![视频](https://img.shields.io/badge/Video-Presentation-F9D371)](https://youtu.be/QLlxulFV0KE)
[![幻灯片](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://drive.google.com/file/d/1t0tthvh_-dd1BvcmokEb-3FUIaEE31DD/view?usp=sharing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on-mscoco)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=bridging-the-gap-between-object-and-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on-1)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-1?p=bridging-the-gap-between-object-and-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on?p=bridging-the-gap-between-object-and-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-object-and-image/open-vocabulary-object-detection-on-lvis-v1-0)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-lvis-v1-0?p=bridging-the-gap-between-object-and-image)

## :rocket: 新闻
* **(2022年10月12日)**
  * 发布交互式 Colab 演示。
* **(2022年9月15日)**
  * 论文被 NeurIPS 2022 接收。
* **(2022年7月7日)**
  * 发布训练和评估代码以及预训练模型。

<hr />

![主要图示](docs/OVD_block_diag.png)
> **<p align="justify"> 摘要:** *现有的开放词汇目标检测器通常通过利用不同形式的弱监督来扩大其词汇量。这有助于在推理时泛化到新的目标。开放词汇检测(OVD)中使用的两种流行的弱监督形式包括预训练的CLIP模型和图像级监督。我们注意到这两种监督模式都没有针对检测任务进行最优对齐：CLIP是通过图像-文本对训练的，缺乏对目标的精确定位，而图像级监督则使用了无法准确指定局部目标区域的启发式方法。在本工作中，我们提出通过对CLIP模型的语言嵌入进行以对象为中心的对齐来解决这个问题。此外，我们使用仅有图像级监督的伪标签过程对目标进行视觉定位，该过程提供高质量的目标建议并有助于在训练期间扩展词汇表。我们通过一个新颖的权重转移函数在上述两种目标对齐策略之间建立桥梁，该函数聚合了它们的互补优势。从本质上讲，所提出的模型试图最小化OVD设置中对象和图像中心表示之间的差距。在COCO基准测试中，我们提出的方法在新类别上达到40.3 AP50，比之前的最佳性能绝对提高了11.9。对于LVIS，我们在稀有类别的掩码AP上超过了最先进的ViLD模型5.0，总体上提高了3.4。* </p>

## 主要贡献

1) **基于区域的知识蒸馏(RKD)** 将以图像为中心的语言表示适应为以对象为中心。
2) **伪图像级监督(PIS)** 使用来自预训练多模态ViTs(MAVL)的弱图像级监督来提高检测器对新类别的泛化能力。
3) **权重转移函数** 有效地结合了上述两个提出的组件。

<hr />

## 安装
代码已在 PyTorch 1.10.0 和 CUDA 11.3 环境下测试。克隆仓库后，请按照 [INSTALL.md](docs/INSTALL.md) 中的步骤进行操作。
我们所有的模型都是使用8个 A100 GPU 训练的。
<hr />

## 演示：创建你自己的自定义检测器
[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](Object_Centric_OVD_Demo.ipynb) 查看我们的交互式 Colab 笔记本演示。使用你自己的类别名称创建自定义检测器。

## 结果
我们展示了面向对象的开放词汇目标检测器的性能，该检测器在开放词汇COCO和LVIS基准数据集上展示了最先进的结果。
对于COCO，基础类别和新类别分别用紫色和绿色表示。
![tSNE图](docs/coco_lvis.jpg)

## 开放词汇COCO结果
### 基线结果
我们在开放词汇COCO基准测试中的结果如下所示。所有结果均使用ResNet-50主干网络。

**开放词汇COCO (基础类别/新类别)**

| 方法 | 主干网络 | AP50 (基础/新) | AP (基础/新) | 下载 |
|:---:|:---:|:---:|:---:|:---:|
| [ViLD](https://arxiv.org/abs/2104.13921) | R50 | 59.5/27.6 | 32.2/11.8 | - |
| [RegionCLIP](https://arxiv.org/abs/2301.02479) | R50 | 56.9/30.7 | 30.7/13.5 | - |
| [OVR-CNN](https://arxiv.org/abs/2108.01809) | R50 | 57.1/27.4 | 31.2/12.2 | - |
| **Ours** | R50 | **63.5/40.3** | **36.5/18.1** | [模型](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EQIsfHZ_FYpIgLGGm2CXFdoBPPykQtJlDHmQHhg4_4L-Yw?e=Hy1Hy1) |

### 消融实验
我们的方法在COCO基准测试上的消融实验结果。

| RKD | PIS | 权重转移 | AP50 (基础/新) | AP (基础/新) |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | | | 61.2/35.4 | 34.1/15.8 |
| | ✓ | | 62.1/36.8 | 35.2/16.4 |
| ✓ | ✓ | | 62.8/38.1 | 35.7/17.2 |
| ✓ | ✓ | ✓ | **63.5/40.3** | **36.5/18.1** |

## LVIS基准测试
### 主要结果
我们在LVIS v1.0验证集上的结果。所有结果均使用ResNet-50主干网络。

| 方法 | APr | APc | APf | AP |
|:---:|:---:|:---:|:---:|:---:|
| [ViLD](https://arxiv.org/abs/2104.13921) | 16.6 | 24.6 | 31.2 | 25.5 |
| [RegionCLIP](https://arxiv.org/abs/2301.02479) | - | - | - | 26.9 |
| [OVR-CNN](https://arxiv.org/abs/2108.01809) | 14.8 | 24.5 | 30.3 | 24.8 |
| **Ours** | **21.6** | **27.8** | **33.1** | **28.9** |

## 训练和评估

### 数据准备
请按照[数据准备指南](docs/DATA_PREPARE.md)准备COCO和LVIS数据集。

### 训练
要训练我们的模型，请运行：
```bash
python train_net.py --num-gpus 8 \
--config-file configs/COCO-OVD/ovd_r50.yaml \
OUTPUT_DIR training_dir/ovd_r50
```

### 评估
要评估训练好的模型，请运行：
```bash
python train_net.py --num-gpus 8 \
--config-file configs/COCO-OVD/ovd_r50.yaml \
--eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

## 引用
如果您发现我们的工作对您的研究有帮助，请考虑引用：
```bibtex
@inproceedings{rasheed2022bridging,
  title={Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection},
  author={Rasheed, Hanoona and Maaz, Muhammad and Khattak, Muhammad Uzair and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## 联系方式
如有任何问题，请联系：
- Hanoona Rasheed [hanoona.rasheed@mbzuai.ac.ae](mailto:hanoona.rasheed@mbzuai.ac.ae)
- Muhammad Maaz [muhammad.maaz@mbzuai.ac.ae](mailto:muhammad.maaz@mbzuai.ac.ae)

## 致谢
- 检测代码基于[Detectron2](https://github.com/facebookresearch/detectron2)
- CLIP实现基于[CLIP](https://github.com/openai/CLIP)