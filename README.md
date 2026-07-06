# BioRetEcho: Biologically-Inspired Low-Light Enhancement

Official PyTorch implementation of **"Biologically-Inspired Low-Light Enhancement with Retinal Echo and Ganglion Cell Feedback"** (Neural Computing and Applications, NCAA-D-26-01277).

## Overview

BioRetEcho is a bio-inspired low-light image enhancement framework that integrates **Retinex physical priors** with **retinal ganglion cell (RGC) spike dynamics** through an encode-feedback-decode architecture. Key features include:

- **Retinal Echo Filter (REF)**: Inspired by bat echolocation, dynamically adjusts weights to enhance dark regions while suppressing overexposure.
- **RGCs-GRU Unit**: Spatio-temporal dual-dimensional module simulating ON/OFF bipolar responses and asynchronous pulse generation.
- **ON/OFF Lateral Inhibition**: Bipolar pulse-driven reflection/illumination (R/S) separation.
- **Self-Generated Pseudo-Label Training**: Eliminates dependency on paired low/normal-light data.
- **Industrial Enhancement (IE) Dataset**: First real-world industrial low-light dataset with 5,000 paired images.

## Architecture

```
Input Low-Light Image
    ↓
[Retinal Echo Filter (REF)]  ←  Bat-inspired dynamic weighting
    ↓
[Encoder]
    ↓
[RGCs-GRU Unit]  ←  Spatio-temporal pulse generation (T=3)
    ↓
[ON/OFF Lateral Inhibition + R/S Separation]
    ↓
[Decoder]  ←  Reflection ⊙ Illumination^γ
    ↓
Enhanced Image
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (for GPU training)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

### LOL Dataset
Download from [LOL Dataset](https://daooshee.github.io/BMVC2018website/).
Organize as:
```
data/LOL/
  ├── train/
  │   ├── low/
  │   └── high/
  └── test/
      ├── low/
      └── high/
```

### ExDark Dataset
Download from [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset).

### IE Dataset (Ours)
Our proposed **Industrial Enhancement (IE) Dataset** with 5,000 paired images is available at:
```
The complete IE dataset and acquisition protocol will be publicly released upon paper acceptance.
```

## Training

### Paired Training (LOL / IE)
```bash
python train.py \
    --dataset LOL \
    --data_root ./data/LOL \
    --paired \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --img_size 400 600 \
    --output_dir ./results \
    --exp_name bioretecho_lol
```

### Unpaired Pseudo-Label Training
```bash
python train.py \
    --dataset IE \
    --data_root ./data/IE \
    --unpaired_normal_dir ./data/normal_images \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --output_dir ./results \
    --exp_name bioretecho_unpaired
```

### Hyperparameter Sensitivity (Ablation)
To reproduce the sensitivity analysis for α and ε:
```bash
for alpha in 0.05 0.10 0.20; do
    for eps in 1e-7 1e-6 1e-5; do
        python train.py \
            --dataset LOL \
            --alpha $alpha \
            --epsilon $eps \
            --exp_name sensitivity_a${alpha}_e${eps}
    done
done
```

## Testing / Inference

```bash
python test.py \
    --dataset LOL \
    --data_root ./data/LOL \
    --checkpoint ./results/bioretecho/checkpoints/best_model.pth \
    --output_dir ./test_results \
    --save_images \
    --save_visuals \
    --compute_flops
```

Results will be saved to `./test_results/` including:
- Enhanced images (`enhanced/`)
- R/S components (`components/`)
- Comparison visualizations (`visuals/`)
- Quantitative metrics (`results.txt`)

## Model Zoo

| Dataset | PSNR (dB) | SSIM | LPIPS | Checkpoint |
|---------|-----------|------|-------|------------|
| LOL     | 21.01     | 0.854| 0.080 | [Download] |
| ExDark  | 23.49     | 0.840| 0.070 | [Download] |
| IE      | 27.55     | 0.970| 0.003 | [Download] |

*LPIPS computed with AlexNet backbone.*

## Efficiency Benchmark

Measured on NVIDIA RTX 3060 (12GB), input size 400×600:

| Method        | Params | FLOPs  | Memory (MB) | Time (ms) | FPS   |
|---------------|--------|--------|-------------|-----------|-------|
| RetinexNet    | 560K   | 300.7G | 420         | 419.8     | 2.4   |
| EnlightenGAN  | 8.6M   | 180.6G | 380         | 78.2      | 12.8  |
| Zero-DCE      | 79K    | 20.8G  | 180         | 18.5      | 54.0  |
| SCI           | 51K    | 1.8G   | 150         | 18.0      | 55.7  |
| RUAS          | 3.4K   | 0.2G   | 65          | 6.5       | 154.0 |
| URetinexNet   | 362K   | 58.3G  | 290         | 35.2      | 28.4  |
| Retinexformer | 1.2M   | 55.4G  | 520         | 180.2     | 5.5   |
| **BioRetEcho**| **221K** | **5.4G** | **210** | **23.0**  | **43.4** |

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{bioretecho2026,
  title={Biologically-Inspired Low-Light Enhancement with Retinal Echo and Ganglion Cell Feedback},
  author={Liu, Xiyang and Song, Jian and others},
  journal={Neural Computing and Applications},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

We thank the anonymous reviewers for their valuable comments and constructive suggestions.
