# Mathematical Notation Reference

**LaTeX-ready formulas with explicit tensor shapes for publication**

---

## Table of Contents
1. [Notation Conventions](#notation-conventions)
2. [Model Architecture](#model-architecture)
3. [Swin Transformer Components](#swin-transformer-components)
4. [CSRF Module](#csrf-module)
5. [MAE Pre-training](#mae-pre-training)
6. [Loss Functions](#loss-functions)
7. [Computational Complexity](#computational-complexity)
8. [Evaluation Metrics](#evaluation-metrics)

---

## 1. Notation Conventions

### General Notation

| Symbol | Description | Dimensions |
|--------|-------------|------------|
| $B$ | Batch size | scalar |
| $H, W$ | Spatial height and width | scalars |
| $D$ | Depth dimension (number of slices) | scalar |
| $C$ | Number of channels | scalar |
| $k$ | Number of input slices (2.5D) | scalar |
| $N$ | Number of tokens/patches | scalar |
| $d$ | Feature dimension | scalar |
| $n_h$ | Number of attention heads | scalar |
| $d_h$ | Head dimension ($d / n_h$) | scalar |

### Tensor Shapes

We use the notation $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$ to denote a 4D tensor with explicit dimensions.

**Example:**
- Input FLAIR volume: $\mathbf{X}_{input} \in \mathbb{R}^{B \times 1 \times 181 \times 217 \times 181}$
- 2.5D input: $\mathbf{X}_{2.5D} \in \mathbb{R}^{B \times k \times H \times W}$
- Feature map: $\mathbf{F} \in \mathbb{R}^{B \times C \times H \times W}$
- Output: $\mathbf{\hat{Y}} \in \mathbb{R}^{B \times 1 \times H \times W}$

---

## 2. Model Architecture

### Overall Pipeline

$$
\mathbf{\hat{Y}} = \text{Decoder}(\text{CSRF}(\mathbf{F}_1, \mathbf{F}_2, \mathbf{F}_3, \mathbf{F}_4))
$$

where:
- $\mathbf{F}_i$ are multi-scale features from the encoder
- $\text{CSRF}$ performs cross-scale residual fusion
- $\text{Decoder}$ produces the final segmentation mask

### 2.5D Input Processing

Given a 3D FLAIR volume $\mathbf{V} \in \mathbb{R}^{H \times W \times D}$, we extract $k$ neighboring slices around the central slice $s_c$:

$$
\mathbf{X}_{2.5D} = [\mathbf{V}_{:,:,s_c - \lfloor k/2 \rfloor}, \ldots, \mathbf{V}_{:,:,s_c}, \ldots, \mathbf{V}_{:,:,s_c + \lfloor k/2 \rfloor}]
$$

Shape: $\mathbf{X}_{2.5D} \in \mathbb{R}^{B \times k \times H \times W}$

### Stem Module

**2.5D Convolutional Stem:**

$$
\mathbf{F}_{stem} = \text{GELU}(\text{BN}(\text{Conv2.5D}(\mathbf{X}_{2.5D})))
$$

where $\text{Conv2.5D}$ is a 2D convolution applied to $k$ input channels:

$$
\text{Conv2.5D}: \mathbb{R}^{B \times k \times H \times W} \rightarrow \mathbb{R}^{B \times C_0 \times \frac{H}{4} \times \frac{W}{4}}
$$

**Parameters:**
- Kernel size: $4 \times 4$
- Stride: $4$
- Output channels: $C_0 = 96$

**LaTeX for Paper:**
```latex
\begin{equation}
\mathbf{F}_{stem} = \text{GELU}(\text{BN}(\text{Conv2.5D}_{4\times4,s=4}(\mathbf{X}_{2.5D})))
\end{equation}
```

Shape transformation: $(B, k, H, W) \rightarrow (B, 96, H/4, W/4)$

---

## 3. Swin Transformer Components

### Patch Partition

Convert feature map to non-overlapping patches:

$$
\mathbf{X}_{patch} = \text{Unfold}(\mathbf{F}, \text{window\_size}=M)
$$

Shape: $(B, C, H, W) \rightarrow (B \cdot N_w, M^2, C)$

where $N_w = \frac{H}{M} \times \frac{W}{M}$ is the number of windows.

### Window-based Multi-Head Self-Attention (W-MSA)

**Attention Mechanism:**

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_h}} + \mathbf{B}\right)\mathbf{V}
$$

where:
- $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{N \times d}$
- $\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{N \times d}$
- $\mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{N \times d}$
- $\mathbf{B} \in \mathbb{R}^{M^2 \times M^2}$ is the relative position bias
- $d_h = d / n_h$ is the head dimension

**Multi-Head Attention:**

$$
\text{MHSA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_h})\mathbf{W}_O
$$

where:

$$
\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^i, \mathbf{X}\mathbf{W}_K^i, \mathbf{X}\mathbf{W}_V^i)
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\text{MHSA}(\mathbf{X}) = \text{Concat}\left(\text{head}_1, \ldots, \text{head}_{n_h}\right)\mathbf{W}_O
\end{equation}

\begin{equation}
\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i\mathbf{K}_i^T}{\sqrt{d_h}} + \mathbf{B}\right)\mathbf{V}_i
\end{equation}
```

### Relative Position Bias

$$
\mathbf{B}(i, j) = \mathbf{B}_{rel}(\Delta x, \Delta y)
$$

where $(\Delta x, \Delta y) = (x_i - x_j, y_i - y_j)$ is the relative position between tokens $i$ and $j$.

**Learnable bias table:**

$$
\mathbf{B}_{rel} \in \mathbb{R}^{(2M-1) \times (2M-1)}
$$

### Shifted Window Multi-Head Self-Attention (SW-MSA)

Shift the window partition by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels:

$$
\hat{\mathbf{X}} = \text{SW-MSA}(\text{LN}(\mathbf{X})) + \mathbf{X}
$$

### Swin Transformer Block

$$
\begin{aligned}
\mathbf{X}^{l+1} &= \text{W-MSA}(\text{LN}(\mathbf{X}^l)) + \mathbf{X}^l \\
\mathbf{X}^{l+2} &= \text{MLP}(\text{LN}(\mathbf{X}^{l+1})) + \mathbf{X}^{l+1}
\end{aligned}
$$

where:

$$
\text{MLP}(\mathbf{X}) = \text{GELU}(\mathbf{X}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

**LaTeX for Paper:**
```latex
\begin{align}
\mathbf{X}^{l+1} &= \text{W-MSA}(\text{LN}(\mathbf{X}^l)) + \mathbf{X}^l \\
\mathbf{X}^{l+2} &= \text{MLP}(\text{LN}(\mathbf{X}^{l+1})) + \mathbf{X}^{l+1}
\end{align}
```

### Patch Merging

Downsample feature maps by concatenating neighboring $2 \times 2$ patches:

$$
\mathbf{X}_{merged} = \text{Linear}(\text{Concat}(\mathbf{X}_{0,0}, \mathbf{X}_{0,1}, \mathbf{X}_{1,0}, \mathbf{X}_{1,1}))
$$

Shape: $(B, H, W, C) \rightarrow (B, H/2, W/2, 2C)$

**LaTeX for Paper:**
```latex
\begin{equation}
\mathbf{X}_{merged} = \mathbf{W}_{merge} \cdot \text{Concat}(\mathbf{X}_{i:i+1,j:j+1}) \in \mathbb{R}^{B \times \frac{H}{2} \times \frac{W}{2} \times 2C}
\end{equation}
```

---

## 4. CSRF Module

### Cross-Scale Residual Fusion

**Formulation:**

Given multi-scale features $\{\mathbf{F}_1, \mathbf{F}_2, \mathbf{F}_3, \mathbf{F}_4\}$ with resolutions:
- $\mathbf{F}_1 \in \mathbb{R}^{B \times C_1 \times H/4 \times W/4}$
- $\mathbf{F}_2 \in \mathbb{R}^{B \times C_2 \times H/8 \times W/8}$
- $\mathbf{F}_3 \in \mathbb{R}^{B \times C_3 \times H/16 \times W/16}$
- $\mathbf{F}_4 \in \mathbb{R}^{B \times C_4 \times H/32 \times W/32}$

**Step 1: Upsampling to target resolution**

$$
\tilde{\mathbf{F}}_i = \text{Upsample}(\mathbf{F}_i, \text{size}=(H_{target}, W_{target}))
$$

where upsampling is bilinear interpolation.

**Step 2: Channel-wise scaling**

$$
\mathbf{F}'_i = \alpha_i \odot \tilde{\mathbf{F}}_i
$$

where:
- $\alpha_i \in \mathbb{R}^{C_i}$ is a learnable per-channel scaling factor
- $\odot$ denotes element-wise multiplication (broadcasting across spatial dims)

**Step 3: Fusion**

$$
\mathbf{F}_{fused} = \sum_{i=1}^{4} \mathbf{W}_i \mathbf{F}'_i
$$

where $\mathbf{W}_i \in \mathbb{R}^{C_{out} \times C_i}$ are $1 \times 1$ convolutions.

**Complete CSRF Equation:**

$$
\boxed{
\mathbf{F}_{fused} = \sum_{i=1}^{4} \text{Conv}_{1\times1}(\alpha_i \odot \text{Upsample}(\mathbf{F}_i))
}
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\mathbf{F}_{fused} = \sum_{i=1}^{4} \mathbf{W}_i \left( \boldsymbol{\alpha}_i \odot \text{Upsample}(\mathbf{F}_i) \right)
\end{equation}

where $\boldsymbol{\alpha}_i \in \mathbb{R}^{C_i}$ are learnable channel-wise scaling factors initialized to 1.
```

### Learnable Scaling Factor

**Initialization:**

$$
\alpha_i^{(0)} = \mathbf{1}_{C_i}
$$

**Update Rule (via backpropagation):**

$$
\alpha_i^{(t+1)} = \alpha_i^{(t)} - \eta \nabla_{\alpha_i} \mathcal{L}
$$

**Constraint (optional clipping):**

$$
\alpha_i \in [0.1, 10.0]
$$

### CSRF Variants (for Ablation)

**1. Scalar Scaling (per-scale):**

$$
\mathbf{F}_{fused}^{scalar} = \sum_{i=1}^{4} \alpha_i \cdot \text{Conv}_{1\times1}(\text{Upsample}(\mathbf{F}_i))
$$

where $\alpha_i \in \mathbb{R}$ is a single scalar per scale.

**2. Per-Channel Scaling (proposed):**

$$
\mathbf{F}_{fused}^{channel} = \sum_{i=1}^{4} \text{Conv}_{1\times1}(\boldsymbol{\alpha}_i \odot \text{Upsample}(\mathbf{F}_i))
$$

where $\boldsymbol{\alpha}_i \in \mathbb{R}^{C_i}$ is a vector of per-channel scales.

**3. No Scaling (baseline):**

$$
\mathbf{F}_{fused}^{baseline} = \sum_{i=1}^{4} \text{Conv}_{1\times1}(\text{Upsample}(\mathbf{F}_i))
$$

---

## 5. MAE Pre-training

### Masked Autoencoder Objective

**Step 1: Random Masking**

Given input patches $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$, randomly mask a subset with ratio $r$:

$$
\mathcal{M} \sim \text{Bernoulli}(r), \quad |\mathcal{M}| = \lfloor r \cdot N \rfloor
$$

**Visible patches:**

$$
\mathbf{X}_{vis} = \{\mathbf{x}_i : i \notin \mathcal{M}\}
$$

**Step 2: Encoder**

$$
\mathbf{Z}_{vis} = \text{Encoder}(\mathbf{X}_{vis})
$$

Shape: $(B, N_{vis}, d) \rightarrow (B, N_{vis}, d)$

**Step 3: Decoder with Mask Tokens**

$$
\mathbf{Z}_{full} = [\mathbf{Z}_{vis}, \mathbf{z}_{mask} \oplus \mathbf{1}_{|\mathcal{M}|}]
$$

where $\mathbf{z}_{mask} \in \mathbb{R}^d$ is a learned mask token.

**Step 4: Reconstruction**

$$
\hat{\mathbf{X}} = \text{Decoder}(\mathbf{Z}_{full})
$$

Shape: $(B, N, d) \rightarrow (B, N, p^2 \cdot c)$

where $p$ is patch size and $c$ is number of channels.

### MAE Loss Function

**Mean Squared Error (MSE) on masked patches only:**

$$
\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|\hat{\mathbf{x}}_i - \mathbf{x}_i\|_2^2
$$

**Normalized MSE (per-patch normalization):**

$$
\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \frac{\|\hat{\mathbf{x}}_i - \mathbf{x}_i\|_2^2}{\|\mathbf{x}_i\|_2^2 + \epsilon}
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|\text{Decoder}(\text{Encoder}(\mathbf{X}_{vis}))_i - \mathbf{x}_i\|_2^2
\end{equation}

where $\mathcal{M}$ is the set of masked patch indices with $|\mathcal{M}| = \lfloor r \cdot N \rfloor$, and $r=0.75$ is the masking ratio.
```

### Bottleneck Feature Reconstruction

**Our Approach:** Reconstruct deep features instead of raw pixels:

$$
\mathcal{L}_{MAE}^{bottleneck} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|\hat{\mathbf{f}}_i - \mathbf{f}_i\|_2^2
$$

where:
- $\mathbf{f}_i$ is the bottleneck feature from a pre-trained encoder
- $\hat{\mathbf{f}}_i$ is the reconstructed feature

**Advantage:** More semantic and less noisy than pixel-level reconstruction.

---

## 6. Loss Functions

### Dice Loss

**Dice Similarity Coefficient:**

$$
\text{DSC}(\mathbf{\hat{Y}}, \mathbf{Y}) = \frac{2 \sum_{i=1}^{N} \hat{y}_i y_i}{\sum_{i=1}^{N} \hat{y}_i^2 + \sum_{i=1}^{N} y_i^2 + \epsilon}
$$

**Dice Loss:**

$$
\mathcal{L}_{Dice} = 1 - \text{DSC}(\mathbf{\hat{Y}}, \mathbf{Y})
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{i=1}^{N} \hat{y}_i y_i + \epsilon}{\sum_{i=1}^{N} \hat{y}_i^2 + \sum_{i=1}^{N} y_i^2 + \epsilon}
\end{equation}
```

where $\epsilon = 1$ is a smoothing constant.

### Binary Cross-Entropy Loss

$$
\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(\hat{y}_i)) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \right]
\end{equation}
```

where $\sigma(\cdot)$ is the sigmoid activation.

### Combined Loss

$$
\boxed{
\mathcal{L}_{total} = \mathcal{L}_{Dice} + \mathcal{L}_{BCE}
}
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Dice} + \lambda_2 \mathcal{L}_{BCE}
\end{equation}
```

where $\lambda_1 = \lambda_2 = 1$ in our experiments.

---

## 7. Computational Complexity

### Parameters

**Total Parameters:**

$$
\text{Params} = \sum_{l=1}^{L} \text{Params}_l
$$

**Convolutional Layer:**

$$
\text{Params}_{conv} = (K_h \times K_w \times C_{in} + 1) \times C_{out}
$$

where $K_h, K_w$ are kernel dimensions, $C_{in}$ input channels, $C_{out}$ output channels, and $+1$ accounts for bias.

**Linear Layer:**

$$
\text{Params}_{linear} = (d_{in} + 1) \times d_{out}
$$

**Multi-Head Self-Attention:**

$$
\text{Params}_{MHSA} = 4 \times d^2 + 4 \times d
$$

for $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$ plus biases.

### FLOPs

**Convolutional Layer:**

$$
\text{FLOPs}_{conv} = 2 \times K_h \times K_w \times C_{in} \times C_{out} \times H_{out} \times W_{out}
$$

**Linear Layer:**

$$
\text{FLOPs}_{linear} = 2 \times d_{in} \times d_{out}
$$

**Self-Attention:**

$$
\text{FLOPs}_{attn} = 2 \times N \times d^2 + 2 \times N^2 \times d
$$

where $N$ is the number of tokens.

**Total Model FLOPs:**

$$
\boxed{
\text{FLOPs}_{total} = \sum_{l=1}^{L} \text{FLOPs}_l = 3.51 \times 10^9
}
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\text{FLOPs}_{total} = \sum_{l=1}^{L} \left( \text{FLOPs}_{conv}^{(l)} + \text{FLOPs}_{attn}^{(l)} + \text{FLOPs}_{linear}^{(l)} \right)
\end{equation}
```

### Memory Complexity

**Training Memory:**

$$
\text{Memory}_{train} = \text{Params} \times 4 + \text{Activations} \times 4 + \text{Gradients} \times 4
$$

(assuming float32, 4 bytes per parameter)

**Inference Memory:**

$$
\text{Memory}_{infer} = \text{Params} \times 4 + \text{Activations}_{peak} \times 4
$$

---

## 8. Evaluation Metrics

### Segmentation Metrics

**Dice Similarity Coefficient (DSC):**

$$
\text{DSC} = \frac{2 |P \cap G|}{|P| + |G|}
$$

where $P$ is predicted mask, $G$ is ground truth.

**Precision:**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Recall (Sensitivity):**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**F1-Score:**

$$
\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Specificity:**

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

### Distance Metrics

**Hausdorff Distance (95th percentile):**

$$
\text{HD95}(P, G) = \max\left\{ h_{95}(P, G), h_{95}(G, P) \right\}
$$

where:

$$
h_{95}(P, G) = \text{percentile}_{95} \left\{ \min_{g \in G} d(p, g) : p \in P \right\}
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\text{HD95}(P, G) = \max\left\{ \text{percentile}_{95}(h(P, G)), \text{percentile}_{95}(h(G, P)) \right\}
\end{equation}
```

### Lesion-wise Metrics

**Lesion F1-Score:**

$$
\text{Lesion-F1} = \frac{2 \times TP_L}{2 \times TP_L + FP_L + FN_L}
$$

where $TP_L, FP_L, FN_L$ are lesion-level true positives, false positives, and false negatives.

**Lesion Detection Criterion:**

A predicted lesion matches a GT lesion if:

$$
\text{IoU}(P_i, G_j) = \frac{|P_i \cap G_j|}{|P_i \cup G_j|} \geq \tau
$$

where $\tau = 0.1$ is the IoU threshold.

### Calibration Metrics

**Expected Calibration Error (ECE):**

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where:
- $B_m$ is the $m$-th confidence bin
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{1}(\hat{y}_i = y_i)$ is accuracy in bin $m$
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$ is average confidence in bin $m$

**LaTeX for Paper:**
```latex
\begin{equation}
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{1}(\hat{y}_i = y_i) - \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i \right|
\end{equation}
```

### Statistical Tests

**Paired t-test:**

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}}
$$

where:
- $\bar{d}$ is mean difference
- $s_d$ is standard deviation of differences
- $n$ is number of pairs

**Cohen's d Effect Size:**

$$
d = \frac{\mu_1 - \mu_2}{\sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}}
$$

**95% Confidence Interval:**

$$
\text{CI}_{95\%} = \bar{x} \pm t_{0.025, n-1} \times \frac{s}{\sqrt{n}}
$$

**LaTeX for Paper:**
```latex
\begin{equation}
\text{CI}_{95\%} = \bar{x} \pm t_{\alpha/2, n-1} \times \frac{s}{\sqrt{n}}
\end{equation}
```

---

## Quick Reference: Copy-Paste LaTeX

### Model Architecture

```latex
% 2.5D Input
\mathbf{X}_{2.5D} \in \mathbb{R}^{B \times k \times H \times W}

% Stem Module
\mathbf{F}_{stem} = \text{GELU}(\text{BN}(\text{Conv2.5D}_{4\times4,s=4}(\mathbf{X}_{2.5D})))

% Swin Block
\begin{align}
\mathbf{X}^{l+1} &= \text{W-MSA}(\text{LN}(\mathbf{X}^l)) + \mathbf{X}^l \\
\mathbf{X}^{l+2} &= \text{MLP}(\text{LN}(\mathbf{X}^{l+1})) + \mathbf{X}^{l+1}
\end{align}

% CSRF Module
\mathbf{F}_{fused} = \sum_{i=1}^{4} \mathbf{W}_i \left( \boldsymbol{\alpha}_i \odot \text{Upsample}(\mathbf{F}_i) \right)

% MAE Loss
\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|\hat{\mathbf{x}}_i - \mathbf{x}_i\|_2^2

% Combined Loss
\mathcal{L}_{total} = \mathcal{L}_{Dice} + \mathcal{L}_{BCE}
```

---

**Last Updated**: November 4, 2025
