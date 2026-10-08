# CIFAR-10 Image Classification

A learning-focused machine learning project for experimenting with Convolutional Neural Networks (CNNs) on the CIFAR-10 image classification dataset using PyTorch.

The main purpose of this repository is not to achieve state-of-the-art accuracy, but to understand how changes to CNN architecture, model capacity, and optimization affect training and generalization.

---

## 1. Project Overview

This project started by adapting a CNN architecture previously used for MNIST to the more complex CIFAR-10 dataset.

Unlike MNIST, CIFAR-10 contains RGB images with significantly more visual variation. The increased complexity provides an opportunity to study how a neural network responds when its capacity is increased through:

* Network depth
* Number of feature channels (width)
* Learning rate
* Optimizer selection
* Training duration
* Validation performance

The experiments are currently intentionally kept relatively small because training is being performed on CPU hardware. More computationally expensive experiments will eventually be moved to Google Colab.

---

## 2. Dataset

The project uses the CIFAR-10 dataset.

CIFAR-10 contains:

* 10 classes
* 50,000 training images
* 10,000 test images
* RGB images
* Image resolution: `32 × 32`
* Input tensor shape for a batch of 64: `(64, 3, 32, 32)`

The ten classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The training data is further divided into:

* Training set: 45,000 images
* Validation set: 5,000 images

The test set remains separate and is used only for final evaluation.

The dataset is downloaded automatically by the PyTorch dataset loader and is not stored in this repository.

---

## 3. Repository Structure

The current project intentionally keeps the implementation relatively simple.

```text
CIFAR-10/
│
├── main.py
│
├── models/
│   └── ...
│
├── README.md
│
└── .gitignore
```

### `main.py`

At the current stage, `main.py` contains the complete experiment pipeline:

1. Dataset loading
2. Training/validation split
3. Data loaders
4. Model creation
5. Training
6. Validation
7. Testing
8. Training statistics

Training and testing have not been separated into different scripts yet.

This is intentional. Since the current experiments are relatively small and the available hardware is CPU-based, separating the pipeline into multiple modules would add structure without providing much practical benefit at this stage.

As the project becomes more complicated, the code can be refactored into separate modules.

---

## 4. Input and Output

The network receives batches with the following shape:

```text
Input:
(64, 3, 32, 32)
```

where:

```text
64  = batch size
3   = RGB channels
32  = image height
32  = image width
```

The final classifier produces:

```text
Output:
(64, 10)
```

Each of the 10 output values corresponds to one CIFAR-10 class.

---

# 5. Experiments

The experiments are being performed incrementally.

Rather than immediately using a large modern architecture, the project starts with a simple CNN and changes one major factor at a time.

The current progression is:

```text
Initial CNN
    ↓
Increase Depth
    ↓
Increase Width
    ↓
Change Learning Rate
    ↓
Change Optimizer
```

This allows the effect of each change to be observed individually.

---

# 6. Experiment 1 — Initial CNN

The first model was based on the CNN previously used for MNIST, adapted for three-channel CIFAR-10 input.

### Architecture

```text
Input
(3, 32, 32)
    ↓
Conv2d
3 → 32
    ↓
ReLU
    ↓
MaxPool
    ↓
Conv2d
32 → 64
    ↓
ReLU
    ↓
MaxPool
    ↓
Flatten
    ↓
Linear
→ 10
```

The important difference from MNIST is that the input now contains three color channels and the image resolution is `32 × 32`.

### Results

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc |
| ----: | ---------: | --------: | ---------: | --------: |
|     1 |     1.7206 |    38.67% |     1.5543 |    44.46% |
|     2 |     1.2936 |    54.63% |     1.4139 |    50.16% |
|     3 |     1.1068 |    61.60% |     1.9776 |    44.36% |
|     4 |     1.0048 |    65.17% |     1.3826 |    56.34% |
|     5 |     0.9309 |    67.78% |     2.3883 |    36.36% |
|     6 |     0.8860 |    69.37% |     1.2281 |    59.92% |
|     7 |     0.8258 |    71.54% |     1.3220 |    59.10% |
|     8 |     0.7861 |    72.79% |     1.1191 |    62.04% |
|     9 |     0.7463 |    74.43% |     1.2201 |    61.00% |
|    10 |     0.7096 |    75.49% |     1.2883 |    60.20% |

**Test Accuracy: 61.04%**

### Observation

The same general CNN structure that performed well on MNIST performed considerably worse on CIFAR-10.

This suggests that CIFAR-10 requires a model with greater representational capacity because the images contain significantly more complex visual patterns.

---

# 7. Experiment 2 — Increasing Depth

The next experiment added another convolutional layer without increasing the number of channels in the second block.

### Architecture

```text
Input
(3, 32, 32)
    ↓
Conv1
3 → 32
    ↓
ReLU
    ↓
MaxPool
(32, 16, 16)
    ↓
Conv2
32 → 64
    ↓
ReLU
    ↓
Conv3
64 → 64
    ↓
ReLU
    ↓
MaxPool
(64, 8, 8)
    ↓
Flatten
    ↓
Linear
→ 10
```

The main architectural change was:

```text
Before:

Conv2 → ReLU → Pool

After:

Conv2 → ReLU → Conv3 → ReLU → Pool
```

The goal was to allow the network to perform another feature transformation before spatial information was discarded by pooling.

### Results

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc |
| ----: | ---------: | --------: | ---------: | --------: |
|     1 |     1.8564 |    32.67% |     1.9963 |    32.68% |
|     2 |     1.3838 |    50.68% |     1.5000 |    44.94% |
|     3 |     1.1685 |    58.91% |     1.4012 |    49.62% |
|     4 |     1.0238 |    64.17% |     1.3358 |    55.12% |
|     5 |     0.9206 |    67.90% |     1.1059 |    61.30% |
|     6 |     0.8393 |    70.87% |     1.2036 |    59.62% |
|     7 |     0.7724 |    73.33% |     1.3341 |    58.60% |
|     8 |     0.7136 |    75.21% |     1.4118 |    53.68% |
|     9 |     0.6676 |    76.88% |     1.1402 |    63.20% |
|    10 |     0.6164 |    78.29% |     1.0876 |    64.94% |

**Test Accuracy: 64.31%**

### Observation

Adding depth improved the test accuracy:

```text
61.04% → 64.31%
```

The model also achieved higher training accuracy.

However, validation accuracy continued to fluctuate substantially, indicating that simply adding depth does not completely solve the generalization problem.

---

# 8. Experiment 3 — Increasing Width

The third experiment increased the number of feature maps.

Instead of:

```text
32 → 64 → 64
```

the model used:

```text
32 → 128 → 128
```

### Architecture

```text
Input
(3, 32, 32)
    ↓
Conv1
3 → 64
    ↓
Pool
(64, 16, 16)
    ↓
Conv2
64 → 128
    ↓
Conv3
128 → 128
    ↓
Pool
(128, 8, 8)
    ↓
Flatten
8192
    ↓
Linear
→ 10
```

### Why increase width?

A convolutional channel can be viewed loosely as a learned feature detector.

For example:

```text
Channel 1 → edge pattern
Channel 2 → another edge pattern
Channel 3 → color pattern
Channel 4 → texture
...
```

Increasing the number of channels gives the network more feature maps with which to represent different patterns.

This is different from increasing depth.

### Depth vs Width

**Depth**

```text
Conv → Conv → Conv
```

provides more sequential feature transformations.

**Width**

```text
32 → 64 → 128
```

provides more feature maps at each transformation.

### Results

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc |
| ----: | ---------: | --------: | ---------: | --------: |
|     1 |     1.8295 |    34.18% |     2.3150 |    29.32% |
|     2 |     1.3497 |    52.05% |     1.6311 |    40.06% |
|     3 |     1.1155 |    60.77% |     1.3459 |    52.98% |
|     4 |     0.9649 |    66.31% |     1.7342 |    42.20% |
|     5 |     0.8541 |    70.46% |     1.3448 |    55.34% |
|     6 |     0.7582 |    73.83% |     1.6431 |    50.50% |
|     7 |     0.6800 |    76.57% |     1.0971 |    63.82% |
|     8 |     0.6044 |    78.88% |     1.7815 |    49.82% |
|     9 |     0.5460 |    81.10% |     1.2785 |    61.16% |
|    10 |     0.4773 |    83.36% |     1.4601 |    59.04% |

**Test Accuracy: 59.10%**

### Observation

Increasing width substantially improved training performance:

```text
Training accuracy:
78.29% → 83.36%
```

However, test accuracy decreased:

```text
64.31% → 59.10%
```

This indicates that the additional capacity allowed the model to fit the training data better without improving generalization.

The model therefore appears to be showing stronger overfitting.

---

# 9. Experiment 4 — Lower Learning Rate

The next experiment kept the wider architecture but changed the optimizer learning rate.

### Previous configuration

```python
optimizer = optim.SGD(lr=0.1)
```

### New configuration

```python
optimizer = optim.SGD(lr=0.01)
```

### Results

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc |
| ----: | ---------: | --------: | ---------: | --------: |
|     1 |     2.1005 |    23.77% |     1.9724 |    28.66% |
|     2 |     1.8078 |    36.55% |     2.2753 |    24.08% |
|     3 |     1.6272 |    42.64% |     1.9405 |    32.10% |
|     4 |     1.5164 |    46.63% |     1.5779 |    41.72% |
|     5 |     1.4443 |    49.06% |     1.4595 |    46.46% |
|     6 |     1.3664 |    51.48% |     1.4903 |    46.40% |
|     7 |     1.2993 |    53.98% |     1.3531 |    51.12% |
|     8 |     1.2412 |    56.17% |     1.5356 |    46.18% |
|     9 |     1.1924 |    57.98% |     1.5504 |    47.14% |
|    10 |     1.1473 |    59.79% |     1.5364 |    47.42% |

**Test Accuracy: 48.17%**

**Total training time: 1186.60 seconds**

### Observation

Reducing the learning rate from `0.1` to `0.01` resulted in substantially slower learning.

After 10 epochs, the model had not reached the training performance of the previous configuration.

Because each epoch already takes a significant amount of time on CPU, running substantially more epochs to determine whether the smaller learning rate would eventually converge was not practical at this stage.

This experiment therefore does not establish that `lr=0.01` is inherently worse. It demonstrates that, under the current 10-epoch CPU-limited experiment, the model did not have enough training time to reach comparable performance.

---

# 10. Experiment 5 — Adam Optimizer

The next experiment returned to the wider CNN and changed the optimizer from SGD to Adam.

### Configuration

```python
optimizer = optim.Adam(lr=0.0001)
```

### Results

| Epoch | Train Loss | Train Acc | Valid Loss | Valid Acc |
| ----: | ---------: | --------: | ---------: | --------: |
|     1 |     1.4576 |    47.89% |     1.2037 |    57.26% |
|     2 |     1.0654 |    62.61% |     0.9838 |    65.44% |
|     3 |     0.8938 |    68.89% |     0.9161 |    67.88% |
|     4 |     0.7810 |    72.91% |     0.8198 |    71.30% |
|     5 |     0.6847 |    76.19% |     0.8039 |    72.14% |
|     6 |     0.6133 |    78.74% |     0.8028 |    72.30% |
|     7 |     0.5433 |    81.18% |     0.8255 |    71.90% |
|     8 |     0.4832 |    83.23% |     0.8484 |    73.06% |
|     9 |     0.4316 |    85.10% |     0.8938 |    72.16% |
|    10 |     0.3854 |    86.49% |     0.9186 |    72.56% |

**Total training time: 1166.07 seconds**

**Test Accuracy: 71.80%**

### Observation

Adam produced a substantial improvement compared with the previous SGD experiments.

The final results were:

```text
Training accuracy: 86.49%
Validation accuracy: 72.56%
Test accuracy:       71.80%
```

Compared with the previous best test accuracy:

```text
64.31% → 71.80%
```

This is currently the best result in the project.

An important observation is that validation accuracy reached approximately 73% while training accuracy continued increasing toward 86%. This indicates that the model is still fitting the training set more strongly than the validation set.

---

# 11. Current Results

The experiments so far can be summarized as follows:

| Experiment       | Main Change         | Test Accuracy |
| ---------------- | ------------------- | ------------: |
| V1               | Initial CNN         |    **61.04%** |
| V2               | Increased depth     |    **64.31%** |
| V3               | Increased width     |    **59.10%** |
| V3 + SGD LR 0.01 | Lower learning rate |    **48.17%** |
| V3 + Adam        | Adam, LR 0.0001     |    **71.80%** |

Current best result:

> **71.80% test accuracy**

using the wider CNN trained with Adam at `lr = 0.0001`.

---

# 12. What Has Been Learned So Far

The experiments demonstrate several important CNN training concepts.

### Increasing depth

Adding another convolutional layer improved the model:

```text
61.04% → 64.31%
```

This suggests that additional feature transformations were useful for the more complex CIFAR-10 images.

### Increasing width

Increasing the number of channels substantially increased training accuracy but reduced test accuracy.

This demonstrates that:

> More model capacity does not automatically produce better generalization.

### Learning rate

A smaller learning rate can require substantially more training iterations to converge.

Because the current hardware is CPU-based, the experiment was limited to 10 epochs and therefore cannot conclusively determine the long-term performance of `lr=0.01`.

### Optimizer

Changing from SGD to Adam produced the largest improvement so far:

```text
64.31% → 71.80%
```

This demonstrates that optimization strategy can have a significant effect even when the model architecture remains largely unchanged.

---

# 13. Hardware Considerations

Training is currently performed on an older Intel i5 CPU.

As a result:

* Individual epochs can take several minutes.
* Large architecture experiments are expensive.
* Running 20–30+ epochs for every experiment is currently impractical.
* Hyperparameter searches are intentionally limited.
* Experiments are selected to isolate important concepts rather than maximize benchmark accuracy.

This hardware limitation is also part of the reason the current implementation keeps training, validation, and testing inside `main.py`.

For future, more computationally expensive experiments, training will be moved to Google Colab.

---

# 14. Future Experiments

The project will gradually move toward more capable architectures and training techniques.

Possible future experiments include:

* Learning-rate decay
* Early stopping
* Better regularization
* Batch normalization
* Data augmentation
* ResNet
* Vision Transformer (ViT)
* Smaller Vision Transformer architectures such as TinyViT
* Comparison between CNN and Transformer-based architectures

The goal is to introduce these techniques progressively and understand why they work rather than immediately using a large pretrained architecture.

---

# 15. Google Colab

This repository is designed to eventually support training on Google Colab when local CPU resources become insufficient.

The intended workflow is:

```text
Local development
       ↓
      Git
       ↓
    GitHub
       ↓
 Google Colab
       ↓
More computationally expensive experiments
```

The dataset itself does not need to be committed to GitHub. The training code can download CIFAR-10 when executed in the new environment.

---

# 16. Installation

Create a Python environment and install the required packages.

At minimum, this project requires:

```text
Python
PyTorch
Torchvision
```

The exact package versions should be recorded in `requirements.txt` as the project environment becomes more stable.

---

# 17. Running the Project

The current experiment pipeline is launched through:

```bash
python main.py
```

`main.py` currently handles the complete process:

```text
Load CIFAR-10
      ↓
Create train/validation split
      ↓
Create DataLoaders
      ↓
Create model
      ↓
Train
      ↓
Validate
      ↓
Test
      ↓
Display results
```

As the project grows, this structure may eventually be refactored into separate training, evaluation, configuration, and model modules.

For now, keeping the pipeline in one file is intentional.

---

# 18. Project Philosophy

This repository is primarily a learning project.

The emphasis is on understanding:

```text
Architecture
     ↓
Capacity
     ↓
Optimization
     ↓
Training behaviour
     ↓
Generalization
```

rather than simply obtaining the highest possible CIFAR-10 accuracy.

Each experiment attempts to answer a specific question about neural-network behaviour before moving to a more complicated model.

---

## Status

**Current best test accuracy: 71.80%**

**Best model so far:** Wider CNN + Adam

**Current training environment:** CPU

**Next major direction:** Improve training efficiency and investigate learning-rate scheduling / regularization before moving to substantially more complex architectures.
