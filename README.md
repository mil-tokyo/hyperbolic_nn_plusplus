# Hyperbolic Neural Networks++
This repository is the official pytorch implementation of [**Hyperbolic Neural Networks++**](https://openreview.net/forum?id=Ec85b0tUwbA), accepted by ICLR2021, aiming to reproduce the results in Section 4.3 out of three main experiments condcted in the paper.


### Requirements
- python: ^3.6
- cuda: ^9
- pytorch: 1.6
- geoopt: 0.3

This repository is built upon the [fairseq](https://github.com/pytorch/fairseq) library, and inherits [geoopt](https://github.com/geoopt/geoopt) library for the computation on the Poincaré ball model.
Please read the README at first to learn how to use fairseq.

## 1. Data Preparation

Basically, this repository can deal with any kind of fairseq-type datasets for translation tasks. 
The following shows one specific example to prepare WMT'17 English to German translation dataset.

### Downloading Data
```bash
mkdir -p /PATH/TO/DATA/scripts
cp examples/translation/prepare-wmt14en2de.sh /PATH/TO/DATA/scripts/
cd /PATH/TO/DATA/scripts
bash prepare-wmt14en2de.sh 
```

Then you can download the dataset in `/PATH/TO/DATA/wmt17_en_de`.

### Preprocessing
```bash
cd bin/preprocess
bash preprocess_wmt17_en_de.sh /PATH/TO/DATA/wmt17_en_de
```

## 2. Training

To train a model, run:

```bash
cd bin/train
bash train_p_fconv_wmt17_en_de.sh N_GPUS GPU_IDS SCRATCH \
    /PATH/TO/DATA/wmt17_en_de/tokenized /PATH/TO/CHECKPOINT \
    DIM DROPOUT WARMUP_STEP MAX_STEP
```
- `N_GPUS` : the number of GPUs you use to train a model, e.g., `4`.
- `GPU_IDS`: a comma separated GPU id list, e.g., `0,1,2,3`.
- `SCRATCH`: a specifier to determine whether to train from scratch `-s` or to resume a previous checkpoint `-r`.
- `DIM`    : dimensionality of feature vectors, e.g., `16`.
- `DROPOUT`: dropout probability, e.g., `0.0`.
- `WARMUP_STEP`: the number of warmup steps, e.g., `4000`.
- `MAX_STEP`: the number of trainig iterations, e.g., `100000`.

## 3. Inference

To verify the performance of a trained model, run:

```bash
cd bin/generate
bash generate_p_fconv_wmt17_en_de.sh GPU_ID \
    /PATH/TO/DATA/wmt17_en_de/tokenized /PATH/TO/CHECKPOINT DIM
```

Then you can get translated sentences and the BLEU score for the test dataset.

## Citation
```
@inproceedings{
shimizu2021hyperbolic,
title={Hyperbolic Neural Networks++},
author={Ryohei Shimizu and YUSUKE Mukuta and Tatsuya Harada},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=Ec85b0tUwbA}
}
```
