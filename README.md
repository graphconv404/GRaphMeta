# GraphMeta

## Abstract
No-reference image quality assessment (NR-IQA) aims to predict perceptual quality in alignment with the human visual system (HVS), yet existing methods face challenges in capturing long-range dependencies across distortion types and levels while preserving content fidelity during preprocessing. This paper presents a perceptually-driven NR-IQA framework that integrates meta-learning, graph representation learning, and multi-scale feature fusion to address these limitations. First, a meta-learning paradigm is employed to pre-train a self-calibrated convolutional backbone, which adaptively models spatial and channel-wise dependencies across scales, thereby enhancing the extraction of distortion-aware features while mitigating information loss caused by fixed-input preprocessing. Second, a graph representation learning module is introduced to explicitly encode the hierarchical relationships among distortion types, distortion levels, and image content. Nodes in the graph correspond to distorted images, while edges capture inter-sample similarities; these are jointly optimized via a graph convolutional network under dual supervision from a triplet-based distortion-type discriminator and a probabilistic distortion-level regressor that accounts for content-induced uncertainty. Extensive experiments on four benchmark datasets demonstrate that our method achieves better performance, with average SROCC and PLCC improvements of 3.6-36.6\% over hand-crafted feature-based methods and consistent gains over deep learning-based approaches. Ablation studies and visualizations confirm that the proposed components collectively yield a more discriminative and generalizable distortion representation, closely mirroring human perceptual judgments.
## Preparation
This project is run on GPU (NVIDIA A6000) with CUDA 12.6.

1. [Optional but recommended] Create a env environment using miniconda, run:
    ~~~
    conda create --name GraphMeta python ==3.7.7
    source activate GraphMeta
    ~~~
2. Install pytorch 1.5.0:
    ~~~
    conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
    ~~~
3. Install required packages by running:
    ~~~
    pip install -r requirements.txt
    ~~~
4. Download [Kadis700k](http://database.mmsp-kn.de/kadid-10k-database.html) dataset and generate the distorted images according to the official guidance, then restore them in `./data/kadis700k/dist_imgs`.
Doneload [LIVE](http://live.ece.utexas.edu/research/Quality/), [LIVEC](http://live.ece.utexas.edu/research/Quality/), [KonIQ10k](http://database.mmsp-kn.de/koniq-10k-database.html) and [CSIQ](https://computervisiononline.com/dataset/1105138666) in `./data/databaserelease2`, `./data/ChallengeDB`, `./data/KonIQ` and `./data/csiq` respectively.

## Usages
When pretraining GraphIQA model from scratch on kadid10k dataset, you need to run:
~~~
python pretrain.py --dataset kadid-P
~~~
When pretraining GraphIQA model from scratch on kadis700k dataset, you need to run:
~~~
python pretrain.py --dataset kadis-P
~~~
Some available options:
* `--dataset`: Training and testing dataset, support datasets: kadid-P | kadis-P
* `--gpus`: The number of required gpus. For pretraining on kadis700k dataset, up to 4 gpus are needed.
* `--gpu_ids`: Visible devices.
* `--bs`: Batch size.
* `--margin`: Margin for triplet loss.

The pretrained model is released [Model](https://drive.google.com/file/d/1mvnfFC4v7P80KhcN2IWCAdQa3y20B_g_/view?usp=sharing).

When finetuning GraphIQA model on target dataset, you need to run:
~~~
python finetune.py --dataset DATASET --restore --ckpt PATH/TO/PRETRAINED_MODEL --gpus 1 --gpu_ids 0 --bs 32
~~~
Some available options:
* `--dataset`: Training and testing dataset, support datasets: live | livec | koniq-10k | csiq | kadid10k
* `--restore`: Enable to load pretrained model.
* `--ckpt`: Path to pretrained model.
* `--gpus`: The number of required gpus. For finetuning, 1 gpu is enough.
* `--gpu_ids`: Visible devices.
* `--bs`: Batch size.

