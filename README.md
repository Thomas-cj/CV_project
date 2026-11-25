# CNN vs Transformer robustness to corruption investigation

## Project Structure 
```text
CV_project/
├── datasets/
│   ├── __init__.py
│   ├── base.py
│   ├── cifar10.py
│   ├── cifar10_blurred.py
│   └── custom_folder.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── deit_tiny.py
│   └── resnet18.py
├── utils/
│   ├── __init__.py
│   ├── attention_utils.py
│   └── gradcam_utils.py
└── run_experiment.py
```
### Base usage examples

```bash
# DeiT-Tiny on clean CIFAR-10 (automatically resizes to 224x224 & runs Attention Rollout)
python run_experiment.py --model deit_tiny --dataset cifar10

# DeiT-Tiny on Gaussian-blurred CIFAR-10
python run_experiment.py --model deit_tiny --dataset cifar10_blurred

# ResNet-18 on clean CIFAR-10 (automatically uses 32x32 stem & runs Grad-CAM)
python run_experiment.py --model resnet18 --dataset cifar10

# ResNet-18 on Gaussian-blurred CIFAR-10
python run_experiment.py --model resnet18 --dataset cifar10_blurred

```

## Using custom image folder
Some datasets are pytorch native, using custom_folder.py we can create our own dataset from disk to load into our experiments.

Class names and integer labels are inferred directly from folder structure (see below)

### Organize images on disk
Note: Must use this folder structure exactly for inference to work

```text
CV_project/
└── data/
    └── aircraft_vs_ships/
        ├── train/
        │   ├── airplane/          # Assigned label 0
        │   │   ├── img_001.jpg
        │   │   └── img_002.png
        │   └── ship/              # Assigned label 1
        │       ├── boat_01.jpg
        │       └── boat_02.jpeg
        └── test/
            ├── airplane/
            │   └── test_01.jpg
            └── ship/
                └── test_02.jpg
```

### Usage examples

```bash
# 1. Train & evaluate ResNet-18 on clean images
python run_experiment.py \
  --model resnet18 \
  --dataset custom_folder \
  --data-root ./data/aircraft_vs_ships \
  --pretrained \
  --epochs 15

# 2. Evaluate DeiT-Tiny on the same dataset
python run_experiment.py \
  --model deit_tiny \
  --dataset custom_folder \
  --data-root ./data/aircraft_vs_ships \
  --pretrained \
  --epochs 15

# 3. Run with Gaussian blur corruption applied to the test set
python run_experiment.py \
  --model deit_tiny \
  --dataset custom_folder \
  --data-root ./data/aircraft_vs_ships \
  --pretrained \
  --blur \
  --blur-kernel 5 \
  --blur-sigma 2.0
```