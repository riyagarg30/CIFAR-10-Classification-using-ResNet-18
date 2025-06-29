# 🧠 CIFAR-10 Classification with Modified ResNet-18

A deep learning project using a custom **ResNet-18** with **Squeeze-and-Excitation (SE)** blocks, **Mixup & CutMix**, **dropout**, and **advanced augmentations** — trained and evaluated on CIFAR-10, achieving **95.94% test accuracy** under a strict 5M parameter limit.

> **Authors**: Pranav Bhatt, Kevin Mai, Riya Garg  
> 📄 [Read the Report](./report.pdf)

---

## 📦 Files Included

| File / Folder               | Description                                                 |
|----------------------------|-------------------------------------------------------------|
| `resnet18_se_cifar10.ipynb`| Jupyter notebook with full training, validation, inference  |
| `report.pdf`               | Project write-up with architecture, results, and analysis   |
| `cifar_test_nolabel.pkl`   | Unlabeled CIFAR-10 test images (used in inference section)  |
| `prediction_csv/`          | Output CSV file with predicted labels                       |

---

## 🚀 Highlights

- ✅ **95.94% test accuracy** with <5M parameters
- 🔧 ResNet-18 modified with `[3, 3, 4, 3]` block config
- 🔄 **Squeeze-and-Excitation blocks** for channel-wise attention
- 🔀 **Mixup & CutMix** augmentation
- 🧪 Advanced data augmentation pipeline
- 🔥 **Warm-up + cosine LR scheduling**
- 🧼 Inference-ready for `.pkl` test file

---

## 🧪 Setup & Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/dl_project_1_public.git
cd dl_project_1_public
````

### 2. Install Required Packages

```bash
pip install torch torchvision numpy matplotlib wandb torchsummary
```

### 3. Run the Notebook

```bash
jupyter notebook resnet18_se_cifar10.ipynb
```

> 🔑 Replace your `wandb.login(key=...)` key in the notebook if needed.

---

## 📊 Results

| Metric              | Value      |
| ------------------- | ---------- |
| ✅ Test Accuracy     | **95.94%** |
| Validation Accuracy | 90.69%     |
| Training Accuracy   | 73.41%     |

---

## 📤 Inference Output

* Inference is performed on `cifar_test_nolabel.pkl`
* Predictions are saved to:
  `prediction_csv/resnet_experiment_YYYYMMDD_HHMMSS_orig_norm.csv`

---

## 🔄 Data Augmentation Techniques

* ✅ Random crop, flip, rotation
* ✅ Color jitter, sharpness, AutoAugment
* ✅ Random erasing
* ✅ Normalization (ImageNet stats)
* ✅ Mixup & CutMix (50/50 hybrid)

---

## 🧠 Model Architecture

* Conv1: 3×3, 32 filters
* Residual Blocks: `[3, 3, 4, 3]` layers
* SE Block: After convs in each block
* Dropout before final FC
* Loss: Label Smoothing Cross-Entropy
* Optimizer: SGD + Momentum + Weight Decay
* Scheduler: Warm-up (10 epochs) → Cosine Annealing

---

## 📄 References

* He et al. (2015) — [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
* Hu et al. (2017) — [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
* Zhang et al. (2017) — [Mixup Augmentation](https://arxiv.org/abs/1710.09412)

