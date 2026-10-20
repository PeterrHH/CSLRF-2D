# CSLRF-2D

**An Enhanced Continuous Sign Language Recognition Framework Based on 2D Contrastive Predictive Coding Pre-Training**

Pengxuan Huang<sup>1</sup>, Siyu Zhou<sup>2</sup>, Jianzhao Feng<sup>3,*</sup>

<sup>1</sup>Delft University of Technology &nbsp;·&nbsp; <sup>2</sup>Ludwig-Maximilians-Universität München &nbsp;·&nbsp; <sup>3</sup>South China Agricultural University &nbsp;·&nbsp;

IPICE 2026 · Proceedings of SPIE 

<p align="center">
  <img src="assets/framework.png" width="850" alt="CSLRF-2D framework"/>
</p>

---

## Overview

Continuous Sign Language Recognition (CSLR) maps a sign language video to a sequence of glosses. Efficient 2D-CNN pipelines are attractive for deployment, but their ImageNet-initialised backbones have no prior knowledge of how visual cues such as hands and faces move from frame to frame.

**CSLRF-2D** improves spatio-temporal representation learning in a 2D framework with three components:

1. **2D Contrastive Predictive Coding (2D-CPC) pre-training.** A self-supervised stage in which an LSTM context model, conditioned on past spatial feature-map locations, predicts embeddings at future frames *and* neighbouring spatial positions with an InfoNCE objective. Unlike standard CPC, it operates on individual locations of the 7×7 feature map rather than on globally pooled frame features.
2. **Multi-head self-attention.** Inserted between the 1D temporal convolution and the Bi-LSTM decoder to capture long-range temporal dependencies.
3. **Joint CTC + auxiliary InfoNCE objective.** During downstream training, the InfoNCE term is kept as a regulariser (L<sub>total</sub> = L<sub>CTC</sub> + 3 · L<sub>InfoNCE</sub>), so the encoder retains the structure learned in pre-training.

2D-CPC pre-training adds **no inference cost**: only the pre-trained encoder is kept for recognition.

## Results

Word Error Rate (WER, lower is better).

| Method                    | PHOENIX-2014-T Dev | PHOENIX-2014-T Test |  CSL-Daily Dev  | CSL-Daily Test |
| ------------------------- | :----------------: | :-----------------: | :-------------: | :-------------: |
| CorrNet (CVPR 2023)       |       19.79       |        20.83        |      30.60      |      30.10      |
| **CSLRF-2D (ours)** |  **19.10**  |   **20.05**   | **29.14** | **28.70** |

**Ablation of components**

| 2D-CPC | Aux. InfoNCE | Self-Attn | PHOENIX-2014-T Dev / Test |  CSL-Daily Dev / Test  |
| :----: | :----------: | :-------: | :-----------------------: | :---------------------: |
|   –   |      –      |    –    |       19.79 / 20.83       |      30.60 / 30.10      |
|   ✓   |      –      |    –    |       19.43 / 20.57       |      30.08 / 29.84      |
|   ✓   |      ✓      |    –    |       19.34 / 20.48       |      29.97 / 29.74      |
|   ✓   |      ✓      |    ✓    |  **19.10 / 20.05**  | **29.14 / 28.70** |

**Efficiency** (downstream model, single NVIDIA RTX 4090)

| Model    | Params (M) | GFLOPs | Latency (ms) |
| -------- | :--------: | :----: | :----------: |
| CorrNet  |   33.19   | 405.0 |     43.6     |
| CSLRF-2D |   39.49   | 405.0 |     43.6     |

The additional 6.30M parameters come entirely from the self-attention module. See the paper for the full prediction-step, loss-weight and attention-head studies.

## Installation

```bash
git clone https://github.com/PeterrHH/CSLRF-2D.git
cd CSLRF-2D
conda env create -f environment.yaml   # or: pip install -r requirements.txt
```

Requirements:

- Python ≥ 3.8 and PyTorch ≥ 1.13
- [ctcdecode](https://github.com/parlance/ctcdecode) == 0.4 for beam-search decoding (Linux only; [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) works on other platforms)
- *Optional:* `sclite` from [Kaldi](https://github.com/kaldi-asr/kaldi) for detailed WER statistics. Link it with `mkdir ./software && ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`, or set `evaluate_tool: python` in `configs/baseline.yaml` to use the Python evaluator instead.

## Data Preparation

Neither dataset is redistributed in this repository. Please obtain them from the original providers and follow their terms of use.

### PHOENIX-2014-T

German Sign Language weather-forecast videos (25 fps, 210×260): 7,096 / 519 / 642 train / dev / test videos, 1,066 signs, 9 signers.

1. Download the dataset from the official page (free for research use):
   **https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/**
2. Extract it and link it into the project:
   ```bash
   ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T
   ```
3. Generate the gloss dictionary and resize frames to 256×256:
   ```bash
   cd preprocess
   python dataset_preprocess-T.py --process-image --multiprocessing
   ```

### CSL-Daily

Chinese Sign Language videos of daily-life communication (30 fps): 18,401 / 1,077 / 1,176 train / dev / test videos, 2,000 signs, 10 signers.

1. Access is **by request**. Follow the application instructions on the official page (an agreement for academic, non-commercial use is required):
   **https://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/**
2. Once access is granted, extract the data and link it:
   ```bash
   ln -s PATH_TO_DATASET ./dataset/CSL-Daily
   ```
3. Preprocess:
   ```bash
   cd preprocess
   python dataset_preprocess-CSL-Daily.py --process-image --multiprocessing
   ```

Choose the target dataset (`phoenix2014-T` or `CSL-Daily`) in `configs/baseline.yaml`.

## Training

Training has two stages. Hyperparameters follow Section 4.1 of the paper and are set in `configs/`.

**Stage 1: 2D-CPC pre-training** (default: time step *t* = 2, pixel step *p* = 2)

```bash
python main.py --config configs/pretrain.yaml --phase pretrain --device 0
```

**Stage 2: downstream CSLR training** (pre-trained encoder, CTC + 3 · InfoNCE, 4-head self-attention)

```bash
python main.py --config configs/baseline.yaml --phase train --device 0 \
    --pretrained-encoder work_dir/pretrain/encoder.pt
```

Configuration priority: command line > config file > argparse defaults.

## Evaluation

```bash
python main.py --config configs/baseline.yaml --phase test --device 0 \
    --load-weights path/to/checkpoint.pt
```

This reports Dev and Test WER.

## Pretrained Models

| Dataset        | Dev WER | Test WER |                                   Download                                   |
| -------------- | :-----: | :------: | :--------------------------------------------------------------------------: |
| PHOENIX-2014-T | 19.10% |  20.05%  | [link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) |
| CSL-Daily      | 29.14% |  28.70%  |          [link](https://ustc-slr.github.io/datasets/2021_csl_daily/)          |

## Visualisation

Grad-CAM heatmaps of the encoder's spatial attention can be produced with `generate_cam.py`. Compared with CorrNet, CSLRF-2D concentrates activation on the hands and face (paper, Figs. 3–4).

## Project Structure

```
CSLRF-2D/
├── configs/           # dataset and training configs
├── dataset/           # data loader (datasets linked here, not included)
├── modules/           # encoder, 2D-CPC, self-attention, temporal conv, Bi-LSTM
├── preprocess/        # dataset preprocessing scripts
├── utils/             # decoding, optimiser, logging, argument parsing
├── evaluation/        # WER evaluation
├── main.py            # entry point: pretrain / train / test
├── slr_network.py     # CSLRF-2D model
└── seq_scripts.py     # training and evaluation loops
```

## Acknowledgements

This work builds directly on **[CorrNet](https://github.com/hulianyuyy/CorrNet)** (Hu et al., CVPR 2023), which served as our baseline and guide for CSLR, and on its predecessor **[VAC](https://github.com/ycmin95/VAC_CSLR)** (Min et al., ICCV 2021). Our implementation is built on the CorrNet codebase: the correlation-augmented ResNet-18 backbone, 1D temporal convolution, Bi-LSTM decoder, data pipeline and evaluation scripts are adapted from it. The 2D-CPC pre-training, the multi-head self-attention module and the joint CTC + InfoNCE objective are contributions of this work. We thank the CorrNet and VAC authors for releasing their code, and the creators of PHOENIX-2014-T and CSL-Daily for making their datasets available.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{huang2026cslrf2d,
  title     = {An Enhanced Continuous Sign Language Recognition Framework Based on 2D Contrastive Predictive Coding Pre-Training},
  author    = {Huang, Pengxuan and Zhou, Siyu and Feng, Jianzhao},
  booktitle = {Proceedings of SPIE, International Conference on Image Processing, Intelligent Control and Computer Engineering (IPICE 2026)},
  year      = {2026},
  doi       = {DOI}
}
```

Please also cite CorrNet and the datasets:

```bibtex
@inproceedings{hu2023corrnet,
  title     = {Continuous Sign Language Recognition with Correlation Network},
  author    = {Hu, Lianyu and Gao, Liqing and Liu, Zekang and Feng, Wei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2529--2539},
  year      = {2023}
}

@inproceedings{camgoz2018nslt,
  title     = {Neural Sign Language Translation},
  author    = {Camgoz, Necati Cihan and Hadfield, Simon and Koller, Oscar and Ney, Hermann and Bowden, Richard},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}

@inproceedings{zhou2021cslDaily,
  title     = {Improving Sign Language Translation with Monolingual Data by Sign Back-Translation},
  author    = {Zhou, Hao and Zhou, Wengang and Qi, Weizhen and Pu, Junfu and Li, Houqiang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```

## License

Released under the [MIT License](LICENSE). Components adapted from CorrNet remain subject to the terms of the original CorrNet repository.

## Contact

Pengxuan Huang · pacers0406@gmail.com · Issues and pull requests are welcome.
