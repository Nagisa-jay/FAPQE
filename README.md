# FAPQE

This project focuses on facial privacy protection. It uses a diffusion-based framework to generate privacy-protected face images and evaluate their resistance against face recognition attacks.


## Features

- Uses Stable Diffusion as the base model for image generation and perturbation optimization.
- Produces privacy-protected face images and saves them to the result directory.
- Supports local face recognition attack evaluation before and after protection.
- Includes configurable options such as frequency-aware adversarial perturbation, quality conditioning, and progressive conditioning.

## Project Structure

```text
FAPQE/
├─ main.py                         # Entry point
├─ adversarial_optimization.py     # Main adversarial optimization and protection pipeline
├─ dataset.py                      # Dataset loading
├─ tests.py                        # Local attack evaluation and result statistics
├─ utils.py                        # Utility functions
├─ align.py                        # Face alignment logic
├─ attention_control.py            # Attention control logic
├─ requirements.txt                # Dependency list
└─ assets/                         # Data and resource directory
```

## Requirements

It is recommended to use Python 3.10 or above. Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- `torch`
- `torchvision`
- `diffusers`
- `facenet-pytorch`
- `opencv-python`
- `scikit-image`
- `lpips`

## Data Preparation

- Place the source face images in `assets/datasets/test` by default, or specify another path with `--source_dir`.
- Results are saved to `results` by default, or to another directory specified with `--protected_image_dir`.

## Usage

Run the project with the default settings:

```bash
python main.py
```

You can also specify custom arguments, for example:

```bash
python main.py \
  --source_dir assets/datasets/test \
  --protected_image_dir results \
  --target_choice 1
```

The program typically performs the following steps:

1. Loads the Stable Diffusion model and DDIM scheduler.
2. Loads the input image dataset.
3. Runs facial privacy protection and adversarial optimization.
4. Evaluates attack performance on clean and protected images.

## Output

- Protected images are saved under `results/` by default.
- Evaluation metrics are written to `result.txt` in the corresponding output directory.
- The testing pipeline may also generate side-by-side visualizations of clean and protected images.

## Acknowledgment

We thank the authors of the original paper for their valuable research and released code. This project is implemented and extended on top of their work.

## Citation

If you use this project in your work, please also cite the original paper:

```bibtex
@inproceedings{salar2025enhancing,
  title={Enhancing facial privacy protection via weakening diffusion purification},
  author={Salar, Ali and Liu, Qing and Tian, Yingli and Zhao, Guoying},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={8235--8244},
  year={2025}
}
```
