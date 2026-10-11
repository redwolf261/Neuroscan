# NeuroScan 3D Model Architecture Documentation

This document provides a comprehensive technical reference for the neural network architectures, internal feature representations, readout mechanisms, and post-processing consensus pipelines used throughout the **NeuroScan** project for 3D multi-parametric brain tumor segmentation (BraTS).

---

## 1. High-Level Architecture Overview

NeuroScan processes 4-channel multi-parametric 3D MRI volumes:
1. **$T_1$-contrast enhanced ($T_1\text{c}$)**
2. **$T_1$-native ($T_1\text{n}$)**
3. **$T_2$-FLAIR ($T_2\text{f}$)**
4. **$T_2$-weighted ($T_2\text{w}$)**

The network predicts three overlapping tumor sub-compartments:
- **Enhancing Tumor (ET)**: Active, contrast-enhancing tumor tissue.
- **Tumor Core (TC)**: Enhancing tumor + necrotic/non-enhancing core ($\text{ET} \subseteq \text{TC}$).
- **Whole Tumor (WT)**: Tumor core + peritumoral edematous tissue ($\text{TC} \subseteq \text{WT}$).

```mermaid
graph TD
    A["4-Channel 3D MRI Patch<br/>(B, 4, 128, 128, 128)"] --> B["UNet3D Encoder Trunk<br/>(enc1, enc2, enc3, bottleneck)"]
    B --> C["Coarse Context Bottleneck<br/>(256ch @ 16³)"]
    C --> D["Bottleneck-Guided Attention Gate<br/>(AttnGate3D)"]
    B -- "Skip Connections" --> E["Decoder Trunk<br/>(dec3, dec2, dec1)"]
    D --> E
    E --> F["D1 Internal Feature Space<br/>(32ch @ 128³)"]
    F --> G["Multi-Task Readout Heads<br/>(seg_head, evidential, boundary)"]
    F --> H["MRD Residual Readout Head<br/>(Frozen Hyperplane w_R, b_R)"]
    G --> I["Spatial Symmetry Consensus<br/>(8-Fold Multi-Axis TTA)"]
    H --> I
    I --> J["Hierarchical Anatomical Consensus<br/>(ET ⊆ WT Containment)"]
    J --> K["Dual-Zone Precision Gating<br/>(Zone 1: Core, Zone 2: Distant MRD)"]
    K --> L["Final Multi-Compartment Mask<br/>(ET, TC, WT)"]
```

---

## 2. Core Trunk Specifications: `UNet3D_v5`

`UNet3D_v5` is the primary production backbone architecture. It combines a 3D U-Net encoder-decoder structure with:
1. **Deep Supervision** at intermediate decoder resolutions ($D/4$ and $D/2$).
2. **Beta-Evidential Uncertainty Estimation** for epistemic confidence.
3. **Decoupled Boundary Gradient Branching**.
4. **Bottleneck-Conditioned Additive Attention Gating** on the finest skip connection (`enc1`).

### Detailed Layer-by-Layer Architecture

| Stage | Sub-Module / Layer | Operation & Kernel | In Channels | Out Channels | Output Resolution (for $128^3$ Input) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Input** | `x` | Raw MRI voxel grid | 4 | 4 | $128 \times 128 \times 128$ |
| **Encoder 1** | `enc1[0]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 4 | 32 | $128 \times 128 \times 128$ |
| | `enc1[1]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 32 | 32 | $128 \times 128 \times 128$ |
| | `pool1` | MaxPool3d ($2^3$, stride 2) | 32 | 32 | $64 \times 64 \times 64$ |
| **Encoder 2** | `enc2[0]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 32 | 64 | $64 \times 64 \times 64$ |
| | `enc2[1]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 64 | 64 | $64 \times 64 \times 64$ |
| | `pool2` | MaxPool3d ($2^3$, stride 2) | 64 | 64 | $32 \times 32 \times 32$ |
| **Encoder 3** | `enc3[0]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 64 | 128 | $32 \times 32 \times 32$ |
| | `enc3[1]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 128 | 128 | $32 \times 32 \times 32$ |
| | `pool3` | MaxPool3d ($2^3$, stride 2) *(v5)* / `RAP` *(v14)* | 128 | 128 | $16 \times 16 \times 16$ |
| **Bottleneck**| `bottleneck[0]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 128 | 256 | $16 \times 16 \times 16$ |
| | `bottleneck[1]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 256 | 256 | $16 \times 16 \times 16$ |
| **Decoder 3** | `upconv3` | ConvTranspose3d ($2^3$, stride 2) | 256 | 128 | $32 \times 32 \times 32$ |
| | `cat3` | Concat(`upconv3`, `enc3`) | $128+128$ | 256 | $32 \times 32 \times 32$ |
| | `dec3[0]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 256 | 128 | $32 \times 32 \times 32$ |
| | `dec3[1]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 128 | 128 | $32 \times 32 \times 32$ |
| **Decoder 2** | `upconv2` | ConvTranspose3d ($2^3$, stride 2) | 128 | 64 | $64 \times 64 \times 64$ |
| | `cat2` | Concat(`upconv2`, `enc2`) | $64+64$ | 128 | $64 \times 64 \times 64$ |
| | `dec2[0]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 128 | 64 | $64 \times 64 \times 64$ |
| | `dec2[1]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 64 | 64 | $64 \times 64 \times 64$ |
| **Attention** | `attn_gate1` | Additive Attention Gating | $256 + 32$ | 32 | $128 \times 128 \times 128$ |
| **Decoder 1** | `upconv1` | ConvTranspose3d ($2^3$, stride 2) | 64 | 32 | $128 \times 128 \times 128$ |
| | `cat1` | Concat(`upconv1`, `enc1_gated`) | $32+32$ | 64 | $128 \times 128 \times 128$ |
| | `dec1[0]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 64 | 32 | $128 \times 128 \times 128$ |
| | `dec1[1]` | Conv3d ($3^3$, pad 1) + BN + ReLU | 32 | 32 | $128 \times 128 \times 128$ |

---

## 3. Dedicated Mechanisms & Mathematical Formulations

### 3.1 AttentionGate3D (Bottleneck-Conditioned Skip Gating)

Unlike conventional Attention U-Nets where gating signals originate from the immediate preceding decoder stage ($D/2$), `UNet3D_v5` conditions gating on the **deepest bottleneck representation ($16^3$, 256 channels)** to directly guide the finest spatial skip ($128^3$, 32 channels).

$$\mathbf{g} = W_g * \mathbf{x}_{\text{bottleneck}}, \quad \mathbf{g} \in \mathbb{R}^{B \times 16 \times 16 \times 16 \times 16}$$

$$\mathbf{g}_{\text{up}} = \text{TrilinearInterpolate}(\mathbf{g}, \text{size}=\mathbf{x}_{\text{enc1}}), \quad \mathbf{g}_{\text{up}} \in \mathbb{R}^{B \times 16 \times 128 \times 128 \times 128}$$

$$\mathbf{x}_{\text{skip}} = W_x * \mathbf{x}_{\text{enc1}}, \quad \mathbf{x}_{\text{skip}} \in \mathbb{R}^{B \times 16 \times 128 \times 128 \times 128}$$

$$\psi = \sigma\left(W_\psi * \text{ReLU}(\mathbf{g}_{\text{up}} + \mathbf{x}_{\text{skip}})\right), \quad \psi \in [0, 1]^{B \times 1 \times 128 \times 128 \times 128}$$

$$\mathbf{x}_{\text{enc1\_gated}} = \mathbf{x}_{\text{enc1}} \odot \psi$$

- $W_g \in \mathbb{R}^{16 \times 256 \times 1 \times 1 \times 1}$ (bias-free Conv3d)
- $W_x \in \mathbb{R}^{16 \times 32 \times 1 \times 1 \times 1}$ (bias-free Conv3d)
- $W_\psi \in \mathbb{R}^{1 \times 16 \times 1 \times 1 \times 1}$ (Conv3d with bias)

---

### 3.2 Rank-Adaptive Pooling (RAP) in `UNet3D_v14`

In `UNet3D_v14`, `pool3` is replaced by Rank-Adaptive Pooling. Standard MaxPool discards 7 of 8 values per $2 \times 2 \times 2$ window. RAP measures the **Participation Ratio (Effective Rank)** $PR(w)$ of the 8 sub-voxel vectors in channel space:

$$PR(w) = \frac{\left(\text{Tr}(G_w)\right)^2}{\|G_w\|_F^2} = \frac{\left(\sum_{i=1}^8 \lambda_i\right)^2}{\sum_{i=1}^8 \lambda_i^2}, \quad G_w = V_w V_w^T \in \mathbb{R}^{8 \times 8}$$

- When $PR(w) \approx 1$, sub-voxels are collinear (redundant); max-pooling preserves all information.
- When $PR(w) \approx 8$, sub-voxels span mutually orthogonal directions; max-pooling discards essential features.

The adaptive pooling operator is:

$$y_w = (1 - a_w) \cdot \max(x_w) + a_w \cdot g(x_w)$$

$$a_w = \text{clip}\left(\lambda \cdot \sigma\left(s \cdot (PR(w) - \tau)\right), 0, 1\right)$$

- $g(x_w)$: Depthwise $2 \times 2 \times 2$ stride-2 convolution ($128 \times 1 \times 2 \times 2 \times 2$ weights).
- $\tau$: Learnable rank threshold (initialized at 1.45).
- $s$: Learnable sharpness factor ($\text{Softplus}(s_{\text{raw}})$, initialized at 4.0).
- $\lambda$: Learnable mixing scale (initialized at 0.05, with identity-at-init property).

---

## 4. Multi-Task Output Heads

All primary heads branch directly from the final internal feature representation:
$$\mathbf{D}_1 = \text{dec1.relu2\_out} \in \mathbb{R}^{B \times 32 \times D \times H \times W}$$

```mermaid
graph LR
    D1["D1 Feature Trunk<br/>(32 channels @ 128³)"] --> H1["seg_head<br/>Conv3D(32, 3, 1³) + Sigmoid"]
    D1 --> H2["evidential_head<br/>Conv3D(32, 6, 1³)"]
    D1 --> H3["boundary_head<br/>Conv3D(32, 3, 1³) [Detached D1]"]
    D1 --> H4["MRD Linear Head<br/>w_R · D1 + b_R"]
    
    H1 --> O1["Production Probabilities<br/>P_ET, P_TC, P_WT"]
    H2 --> O2["Beta Parameters<br/>α, β (Epistemic Variance)"]
    H3 --> O3["Boundary Logits<br/>Distance-to-Edge Alignment"]
    H4 --> O4["Residual Recovery Map<br/>R(x) for G1 Rescues"]
```

### 1. Primary Segmentation Head (`seg_head`)
$$\mathbf{P}_{\text{prod}} = \sigma\left(W_{\text{seg}} * \mathbf{D}_1 + \mathbf{b}_{\text{seg}}\right) \in [0, 1]^{B \times 3 \times D \times H \times W}$$
- Weight tensor: $W_{\text{seg}} \in \mathbb{R}^{3 \times 32 \times 1 \times 1 \times 1}$
- Bias vector: $\mathbf{b}_{\text{seg}} \in \mathbb{R}^3$
- Channel mapping: Channel 0 = ET, Channel 1 = TC, Channel 2 = WT.

### 2. Evidential Beta Head (`evidential_head`)
Computes conjugate Dirichlet / Beta parameters for evidential uncertainty quantification:
$$\mathbf{z}_{\text{evid}} = W_{\text{evid}} * \mathbf{D}_1 + \mathbf{b}_{\text{evid}} \in \mathbb{R}^{B \times 6 \times D \times H \times W}$$
$$\alpha = \text{Softplus}(\mathbf{z}_{\text{evid}}[:3]) + 1.0, \quad \beta = \text{Softplus}(\mathbf{z}_{\text{evid}}[3:]) + 1.0$$
$$\text{Expected Probability} = \frac{\alpha}{\alpha + \beta}, \quad \text{Epistemic Variance} = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

### 3. Decoupled Boundary Head (`boundary_head`)
Supervises surface normals and boundary sharpness without allowing boundary gradients to destabilize volumetric representations:
$$\mathbf{z}_{\text{bnd}} = W_{\text{bnd}} * \text{detach}(\mathbf{D}_1) + \mathbf{b}_{\text{bnd}} \in \mathbb{R}^{B \times 3 \times D \times H \times W}$$

### 4. Deep Supervision Auxiliary Heads (`aux_head3`, `aux_head2`)
- `aux_head3`: Conv3D(128, 3, 1x1x1) + Sigmoid operating on `dec3` ($32^3$ resolution). Supervised against $4\times$ downsampled ground truth.
- `aux_head2`: Conv3D(64, 3, 1x1x1) + Sigmoid operating on `dec2` ($64^3$ resolution). Supervised against $2\times$ downsampled ground truth.

---

## 5. Novelty: Pairwise Evidence Geometry & Marginal Recovery Decomposition (MRD)

The primary methodological and theoretical novelty of the NeuroScan project is **Pairwise Evidence Geometry (PEG)** and its operationalization via **Marginal Recovery Decomposition (MRD)**.

### 5.1 Theoretical Grounding & The Representation Gap
Standard deep segmentation models collapse the rich representation space of the final decoder trunk ($\mathbf{D}_1 \in \mathbb{R}^{32}$) onto a single hyperplanar decision boundary:
$$\mathbf{P}_{\text{prod}}(\mathbf{x}) = \sigma\left(\mathbf{w}_P^T \mathbf{x} + b_P\right)$$

In multi-parametric brain tumor segmentation, this single hyperplane frequently misses small, faint, or non-contrast-saturating lesion clusters (Group 1 / $G_1$ false negatives). However, empirical analysis of the latent manifold reveals that **the features of these missed lesions are not destroyed by the encoder-decoder trunk**; rather, they are merely separated from the background along a *different direction* in $\mathbb{R}^{32}$ that is orthogonal to $\mathbf{w}_P$.

### 5.2 Marginal Recovery Target Transform $T_{\text{MRD}}(y, p)$
To isolate and supervise this latent residual information without disrupting the calibrated production weights, we formulate a non-linear target decomposition:

$$T_{\text{MRD}}(y, p) = \begin{cases} 
1 & \text{if } y = 1 \text{ and } p < \tau_P \quad (\text{Recoverable } G_1 \text{ False Negative}) \\ 
0 & \text{if } y = 0 \quad (\text{True Negative / Background}) \\ 
0 & \text{if } y = 1 \text{ and } p \ge \tau_P \quad (\text{Production True Positive}) 
\end{cases}$$

Where:
- $y \in \{0, 1\}$ is the voxel ground truth label.
- $p \in [0, 1]$ is the production baseline probability $\mathbf{P}_{\text{prod}}(\mathbf{x})$.
- $\tau_P = 0.50$ is the standard production decision threshold.

### 5.3 Orthogonal Hyperplane Geometry
A frozen linear readout head $(\mathbf{w}_R, b_R)$ is trained on $\mathbf{D}_1$ using $T_{\text{MRD}}$:
$$z_R(\mathbf{x}) = \mathbf{w}_R^T \mathbf{x} + b_R, \quad R(\mathbf{x}) = \sigma\left(z_R(\mathbf{x})\right)$$

The angle between the primary production vector $\mathbf{w}_P$ and the residual recovery vector $\mathbf{w}_R$ is nearly orthogonal:
$$\cos \theta = \frac{\mathbf{w}_P^T \mathbf{w}_R}{\|\mathbf{w}_P\|_2 \|\mathbf{w}_R\|_2} \approx 0.12 - 0.24 \quad \implies \quad \theta \approx 76^\circ - 83^\circ$$

This near-orthogonal geometric relationship proves that the network trunk internally learns distinct sub-manifolds for primary tumor mass and marginal/satellite lesions.

---

## 6. Novelty: Dual-Zone Precision Gating & Hierarchical Consensus (E293)

While $R(\mathbf{x})$ detects $G_1$ missed lesions with high sensitivity, direct unconstrained thresholding of $R(\mathbf{x})$ causes false-positive expansion along tumor boundaries and parenchymal noise. E293 resolves this through a three-stage post-processing consensus:

```mermaid
sequenceDiagram
    participant Input as 4D MRI Patch (x)
    participant Trunk as UNet3D_v5 Trunk
    participant TTA as 8-Fold Spatial Symmetry
    participant Hier as Hierarchical Consensus (ET ⊆ WT)
    participant Res as Frozen MRD Head (w_R, b_R)
    participant Gate as A2 27-Feature Classifier
    participant Out as Calibrated ET Mask

    Input->>Trunk: Forward pass across 8 dihedral flips
    Trunk->>TTA: Symmetrically unflip and average predictions
    TTA->>Hier: Soft probabilities P_ET, P_WT
    Hier->>Hier: Compute Core: (P_ET >= 0.15) & (P_WT >= 0.50)
    TTA->>Res: Evaluate R(x) on D1 internal features
    Res->>Gate: Extract candidate components with R(x) >= 0.9508
    Gate->>Gate: Zone 1 (<= 2mm): Suppress candidates (prevent boundary bleed)
    Gate->>Gate: Zone 2 (> 2mm): Filter via 27 features (Accept if P_A2 >= 0.80)
    Gate->>Out: Fuse Core (Zone 1) + Accepted Isolated Lesions (Zone 2)
```

### 6.1 Spatial Symmetry Consensus (8-Fold Dihedral TTA)
To suppress directional convolutional bias and edge artifacts, inputs undergo 8-fold test-time augmentation spanning the 3D reflection group:
$$\bar{\mathbf{P}}(\mathbf{x}) = \frac{1}{8} \sum_{d \in \{0,1\}} \sum_{h \in \{0,1\}} \sum_{w \in \{0,1\}} \text{Flip}_{d,h,w}\left(\mathcal{M}\left(\text{Flip}_{d,h,w}(\mathbf{x})\right)\right)$$

### 6.2 Hierarchical Anatomical Consensus ($ET \subseteq WT$)
Biologically, the Enhancing Tumor (active neovascularized core) cannot exist in normal, unedematous brain parenchyma outside the Whole Tumor envelope. We enforce this constraint through joint soft thresholding:

$$\text{Core}_{\text{ET}}(\mathbf{x}) = \left(\bar{\mathbf{P}}_{\text{ET}}(\mathbf{x}) \ge \tau_{\text{ET}}^*\right) \land \left(\bar{\mathbf{P}}_{\text{WT}}(\mathbf{x}) \ge \tau_{\text{WT}}^*\right)$$

- Operating parameters (strictly frozen on the 13 Dev validation subjects):
  - $\tau_{\text{ET}}^* = 0.15$ (captures faint, low-contrast tumor boundaries).
  - $\tau_{\text{WT}}^* = 0.50$ (acts as an anatomical containment gate).
- Result: Eliminates $93.4\%$ of isolated false positives that would otherwise arise from lowering $\tau_{\text{ET}}$ to $0.15$.

### 6.3 Dual-Zone Candidate Precision Gating ($A_2$)
For voxels outside $\text{Core}_{\text{ET}}$, candidate connected components $\mathcal{C}_k$ are identified where $R(\mathbf{x}) \ge \tau_R^* = 0.9508$. The spatial domain is partitioned by Euclidean distance to the core:
$$\delta(\mathbf{x}) = \min_{\mathbf{y} \in \text{Core}_{\text{ET}}} \|\mathbf{x} - \mathbf{y}\|_2$$

1. **Zone 1 ($\delta(\mathbf{x}) \le 2.0\text{ mm}$ - Boundary Shell)**:
   - Candidates are **strictly rejected**. All boundary adjustments must originate from the hierarchically calibrated core $\text{Core}_{\text{ET}}$ to prevent boundary dilation.
2. **Zone 2 ($\delta(\mathbf{x}) > 2.0\text{ mm}$ - Distant Parenchyma)**:
   - Candidates represent potential true satellite or multifocal lesions.
   - Each component $\mathcal{C}_k$ is evaluated by a pre-registered **27-feature Logistic Regression classifier ($A_2$)**:
     - *Morphology (6)*: Volume (voxels), surface area, sphericity, compactness, bounding box aspect ratio, principal eigenvalue ratio.
     - *MRI Multi-Modal Intensity (8)*: Mean and variance across $T_1\text{c}, T_1\text{n}, T_2\text{f}, T_2\text{w}$.
     - *Spatial Position (4)*: Centroid $(z, y, x)$ normalized to brain bounding box, distance to ventricles.
     - *Confidence & Uncertainty (5)*: Mean and peak $R(\mathbf{x})$, evidential variance $\sigma^2_{\text{epistemic}}$, entropy.
     - *Contextual Isolation (4)*: Distance to nearest TC boundary, distance to brain surface, local background contrast ratio.
   - Component is accepted if:
     $$P_{\text{accept}}(\mathcal{C}_k) \ge \tau_{\text{accept}} = 0.80$$

### 6.4 E293 Single-Touch Locked Test Validation (125 Subjects)
The complete E293 system was evaluated in a single-touch test on the official 125 held-out validation cohort without any post-hoc adjustments:
- **B0 (Baseline Production, $\tau=0.50$)**: Mean ET Dice = `0.8161`, Global Pooled Dice = `0.8828`
- **B2 (Hierarchical Core Alone)**: Mean ET Dice = **`0.8227`** ($\mathbf{+0.65\%}$, Wilcoxon $p < 0.0001$), Global Pooled Dice = `0.8877`
- **B3 (Complete Integrated System)**: Mean ET Dice = **`0.8224`** ($\mathbf{+0.63\%}$), rescuing $27 / 162$ ($16.7\%$) of completely missed $G_1$ lesions.

---

## 7. Model Checkpoints & Execution Specs

| Model Tag | Checkpoint Path | Architecture Class | Training Loss | Best Mean Dice (Val) |
| :--- | :--- | :--- | :--- | :---: |
| **M1 (Production Base)** | `experiments/exp_e12_eggo_m/e131/runs/E131_v5control_seed0/checkpoints/best.pth` | `UNet3D_v5(4, 3)` | FocalTversky + Evidential + Boundary + Aux | **0.8929** |
| **M2 (Evidence Base)** | `experiments/exp_e12_eggo_m/e141/runs/E141_evidence_seed0/checkpoints/best.pth` | `UNet3D_v5(4, 3)` | Evidence-Weighted FocalTversky | **0.8922** |
| **M3 (RAP Base)** | `experiments/exp_e12_eggo_m/e131/runs/E131_v14_seed0/checkpoints/best.pth` | `UNet3D_v14(4, 3)` | Rank-Adaptive Pooling at Pool3 | **0.8942** |

### Computational Footprint
- **Parameters**: 5.61M (`5,610,691` for `UNet3D_v5`), 5.61M (`5,611,717` for `UNet3D_v14`)
- **Patch Dimension**: $128 \times 128 \times 128$ voxels (isotropic 1.0 mm spacing)
- **VRAM Consumption**: 4.62 GB during training (batch size 1, AMP enabled); 882 MB during evaluation.
- **Inference Speed**: ~0.05 seconds per single forward pass; ~3.4 seconds for 8-fold TTA on an NVIDIA RTX GPU.
