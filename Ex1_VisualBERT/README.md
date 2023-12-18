# VisualBERT:一个简单而高效的视觉和语言 Baseline

论文链接：[VisualBERT: A Simple and Performant Baseline for Vision and Language (arxiv)](https://arxiv.org/abs/1908.03557).

`pytorch_pretrained_bert` 是 HuggingFace 的 Pytorch BERT 的早期克隆的修改版本。

VisualBERT 的核心部分主要通过修改 `modeling.py` 来实现。两个 wrapping model 在`models/model.py `中实现，用于加载不同数据集的代码在  `dataloders` 中，其中 allenlp 的 `Field` 和 `instance` 被广泛用于 wrapping data。

# 依赖关系

## 基础部分

这个存储库的依赖关系类似于 R2C。

如果您不需要自己提取图像特征或在 VCR 上运行，则需要以下基本依赖项（假设您使用的是新的 conda 环境）：

```
conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg


#Please check your cuda version using `nvcc --version` and make sure the cuda version matches the cudatoolkit version.
conda install pytorch torchvision cudatoolkit=XXX -c pytorch


# Below is the way to install allennlp recommended in R2C. But in my experience, directly installing allennlp seems also okay.
pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm
pip install attrdict
pip install pycocotools
pip install commentjson
```

## 提取图像特征部分

这部分只有在您想要在VCR上运行或自己提取图像功能时才安装。

1. pytorch vision 的特殊版本，具有 ROIAlign 层:
```
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d
```
2. [Detectron](https://github.com/facebookresearch/Detectron/)


## 故障排除部分
以下是我在安装这些依赖项时遇到的一些问题：

1. pyyaml version.

当pyyaml版本太高时，Detectron可能会寄（不确定现在是否已修复）。

> 解决方法: install a lower version `pip install pyyaml==3.12`. (https://github.co m/facebookresearch/Detectron/issues/840, https://github.com/pypa/pip/issues/5247)

2. Error when importing torchvision or ROIAlign layer.

> 很可能是 cudatoolkit 和 cuda 版本不匹配。请在 `conda install pytorch torchvision cudatoolkit=XXX -c pytorch` 中指定正确的 cudatoolkit 版本。

3. Segmentation fault when runing ResNet50 detector in VCR.

> 很可能 GCC 版本太低。需要 GCC 版本&gt;= 5.0。`conda install -c psi4 gcc-5` seems to solve the problem. (https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/TROUBLESHOOTING.md)


# 运行代码

假设文件夹 XX 是代码目录的父目录。
```
export PYTHONPATH=$PYTHONPATH:XX/visualbert/
export PYTHONPATH=$PYTHONPATH:XX/

cd XX/visualbert/models/

CUDA_VISIBLE_DEVICES=XXXX python train.py -folder [where you want to save the logs/models] -config XX/visualbert/configs/YYYY
```
在 `visualbert/configs` 中是不同数据集上不同模型的配置。请更改配置中的数据路径 `data_root` 和模型路径 `restore_bin`，这是您想要初始化的模型的路径，以匹配您的本地设置。

## NLVR2

### Prepare Data

将我们预先计算的特性下载到 X_NLVR 文件夹中 ([Train](https://drive.google.com/file/d/1iK9CDfxZ4ejKRWOIItLhD8-sgw78ld7w/view?usp=sharing), [Val](https://drive.google.com/file/d/13rFujBIBr6PLnG5A5i8WJJVT52RPYH9j/view?usp=sharing), [Test-Public](https://drive.google.com/file/d/1RTXZCK_kbFkqOeBnZ5wOAyDSzlmuaKRx/view?usp=sharing))

图像特征来自Detectron的模型(e2e_mask_rcnn_R-101-FPN_2x, model_id: 35861858)。

在命令行中从Google Drive下载链接，请点击 https://github.com/gdrive-org/gdrive

然后下载三个json文件 [the NLVR github page](https://github.com/lil-lab/nlvr/tree/master/nlvr2/data) 到 X_NLVR中

对于COCO预训练，首先将COCO标题注释下载到X_COCO文件夹中。

```
cd X_COCO
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

然后将COCO图像特性下载到X_COCO。 [Train](https://drive.google.com/file/d/1F-LSQhpKleV4nmiKMjQvpHMS2gkbK3bY/view?usp=sharing), [Val](https://drive.google.com/file/d/1cZjPob3YqfM46LaWY3-Ky12claxeXbWi/view?usp=sharing).

### COCO Pre-training
The corresponding config is `visualbert/configs/nlvr2/coco-pre-train.json`. [Model checkpoint](https://drive.google.com/file/d/1QvivVfRsRF518OQSQNaN7aFk6eQ43vP_/view?usp=sharing).

### Task-specific Pre-training
The corresponding config is `visualbert/configs/nlvr2/pre-train.json`. [Model checkpoint](https://drive.google.com/file/d/1Z19G_rAuKn0TQ5Cj-KCcavBKSwBhvxGq/view?usp=sharing).

### Fine-tuning
The corresponding config is `visualbert/configs/nlvr2/fine-tune.json`. [Model checkpoint](https://drive.google.com/file/d/1GCV6woBnWY09JhjtLOXyKUhuFiQz9L5U/view?usp=sharing).



## VQA

### Prepare Data
The image features and VQA data are from Pythia. Assuming the data is stored in X_COCO.

```
cd X_COCO
cd data
wget https://dl.fbaipublicfiles.com/pythia/data/vocabulary_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/imdb.tar.gz
gunzip imdb.tar.gz 
tar -xf imdb.tar

wget https://dl.fbaipublicfiles.com/pythia/features/detectron_fix_100.tar.gz
gunzip detectron_fix_100.tar.gz
tar -xf detectron_fix_100.tar
rm -f detectron_fix_100.tar
```

### COCO Pre-training

The corresponding config is `visualbert/configs/vqa/coco-pre-train.json`. [Model checkpoint](https://drive.google.com/file/d/1tgYovjB6MZZlqdSAOPzB8bZqnFezWNBO/view?usp=sharing).

### Task-specific Pre-training
The corresponding config is `visualbert/configs/vqa/pre-train.json`. [Model checkpoint](https://drive.google.com/file/d/1kuPr187zWxSJbtCbVW87XzInXltM-i9Y/view?usp=sharing).

### Fine-tuning
The corresponding config is `visualbert/configs/vqa/fine-tune.json`. [Model checkpoint](https://drive.google.com/file/d/19FpfLYo3rwv0eybUvfkDMCoivyL4XLqB/view?usp=sharing).



## VCR
### Prepare Data

Download vcr images and annotations.
```
cd X_VCR
wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1annots.zip
wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip
unzip vcr1annots.zip
unzip vcr1images.zip
```
For COCO pre-training, first download raw COCO images:
```
cd X_COCO
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
```
Then download the detection results (boxes and masks) on COCO ([Train](https://drive.google.com/file/d/1lmPiz8dsM0jwJmooVcMRTLa4YGmf_qU_/view?usp=sharing), [Val](https://drive.google.com/file/d/1fVX4TaqcgowoWQTNJ8k3EYxUKRpMJSFL/view?usp=sharing)) from a large detector used when creating VCR dataset to X_COCO.

### COCO Pre-training
The corresponding config is `visualbert/configs/vcr/coco-pre-train.json`. [Model checkpoint](https://drive.google.com/file/d/1pPobkXAL9Evlp7fDjPeXnixtG0O-efoH/view?usp=sharing).

### Task-specific Pre-training
The corresponding config is `visualbert/configs/vcr/pre-train.json`. [Model checkpoint](https://drive.google.com/file/d/1iZ7QUv_jG6E6KNofO0jM5H9ee7nMEuYM/view?usp=sharing).

### Fine-tuning
The corresponding config is `visualbert/configs/vcr/fine-tune.json`. [Model checkpoint](https://drive.google.com/file/d/1z7XSUpPthhBKvgKb0wOcBe2eUNGDmf2o/view?usp=sharing).


## Flickr30K
### Prepare Data

The data processiong follows [VQA-BAN](https://github.com/jnhwkim/ban-vqa).

For COCO pre-training, first download COCO features to a folder X_COCO (script from [VQA-BAN](https://github.com/jnhwkim/ban-vqa)).

```
cd X_COCO
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
wget https://imagecaption.blob.core.windows.net/imagecaption/test2014.zip
wget https://imagecaption.blob.core.windows.net/imagecaption/test2015.zip
unzip trainval.zip
unzip test2014.zip
unzip test2015.zip
rm trainval.zip
rm test2014.zip
rm test2015.zip
```

For fine-tuning on Flickr, follow [VQA-BAN, Flickr30K](https://github.com/jnhwkim/ban-vqa#flickr30k-entities) to process the data files. After pre-processing, put files with suffixes of `_imgid2idx.pkl`, `hdf5`, and the folder `Flickr30kEntities` under X_FLICKR.

### COCO Pre-training
The corresponding config is `visualbert/configs/flickr/coco-pre-train.json`.

### Task-specific Pre-training
The corresponding config is `visualbert/configs/flickr/pre-train.json`.

### Fine-tuning
The corresponding config is `visualbert/configs/flickr/fine-tune.json`.


## 自己提取图像特征
### Extract features using Detectron for NLVR2
Dowload the corresponding config (XXX.yaml) and checkpoint (XXX.pkl) from [Detectron](https://github.com/facebookresearch/Detectron). The model I used is 35861858. 

Download NLVR2 images to a folder X_NLVR_IMAGE (you need to request them from the authors of NLVR2). https://github.com/lil-lab/nlvr/tree/master/nlvr2

Then run:
```
#SET = train/dev/test1
cd visualbert/utils/get_image_features
CUDA_VISIBLE_DEVICES=0 python extract_features_nlvr.py --cfg XXX.yaml --wts XXX.pkl --min_bboxes 150 --max_bboxes 150 --feat_name gpu_0/fc6 --output_dir X_NLVR --image-ext png X_NLVR_IMAGE/SET --no_id --one_giant_file X_NLVR/features_SET_150.th
```

