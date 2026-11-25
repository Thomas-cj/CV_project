Planned structure


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
# DeiT-Tiny on clean CIFAR-10 (automatically resizes to 224x224 & runs Attention Rollout)
python run_experiment.py --model deit_tiny --dataset cifar10

# DeiT-Tiny on Gaussian-blurred CIFAR-10
python run_experiment.py --model deit_tiny --dataset cifar10_blurred

# ResNet-18 on clean CIFAR-10 (automatically uses 32x32 stem & runs Grad-CAM)
python run_experiment.py --model resnet18 --dataset cifar10

# ResNet-18 on Gaussian-blurred CIFAR-10
python run_experiment.py --model resnet18 --dataset cifar10_blurred

```