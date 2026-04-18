# EQ-Reg: A Unified Regularization Framework for Rotation Equivariant Image Restoration
Official implementation.

> **Abstract**: Incorporating transformation symmetry into deep learning has shown notable success in enhancing robustness and generalization. 
> However, existing equivariant approaches often suffer from limited representation accuracy and rely on strict symmetry assumptions that may not hold in practice.
> These limitations pose a significant drawback for image restoration tasks, which demands high accuracy and precise symmetry representation.
> To address these challenges, we propose a rotation-equivariant regularization strategy that adaptively enforces the appropriate symmetry constraints on the data while preserving the network's representational accuracy. 
> Specifically, we introduce EQ-Reg, a regularization term designed to enhance rotation equivariance, which intrinsically extends the insights of both the data-augmentation-based and group-equivariant-based methodologies. 
> This is achieved by applying the spatial rotation and cyclic channel shift of feature maps properties, derived from state-of-the-art equivariant network models, to the formulation of the core loss function in self-supervised learning.
> By enforcing this equivariance regularization across intermediate feature representations, our method enables both convolution-based and Transformer-based models to adaptively learn symmetry priors for all network layers, in a plug-and-play manner. Extensive experiments on multiple low-level vision tasks demonstrate the superior accuracy and generalization capability of our method, outperforming state-of-the-art approaches.

## 📌 Introduction

Rotation equivariance is an important prior for image restoration, especially when the underlying structures exhibit approximate rotational symmetry. Existing approaches often rely on specially designed equivariant architectures, which may suffer from restricted representation flexibility or imperfect matching to real data.

In this work, we propose **EQ-Reg**, a unified regularization framework for learning rotation equivariant priors in image restoration. Instead of enforcing strict equivariance through architectural design, EQ-Reg introduces an explicit regularization objective that encourages the model to capture the appropriate degree of rotational equivariance directly from data. The framework is general and can be integrated into different restoration backbones, including both CNN-based and Transformer-based models.

## 🖼️ Framework Overview

<p align="center">
  <img src="image/framwork.png" width="85%">
</p>

## 📂 Experiments
We provide example implementations of **EQ-Reg** on two representative Transformer-based image restoration architectures: **Swin Transformer** and **Restormer**. The corresponding official projects are available at [SwinIR](https://github.com/JingyunLiang/SwinIR) and [Restormer](https://github.com/swz30/Restormer). :contentReference[oaicite:0]{index=0}

### Swin Transformer Example

For experiments based on Swin Transformer, we provide a ready-to-use implementation under `./KAIR_EQ/`. The required conda environment file is included in this directory as `environment.yml`. After creating the environment, all training and testing commands can be found in `run.sh`.

### Restormer Example

For experiments based on Restormer, we recommend following the installation instructions in the official [Restormer repository](https://github.com/swz30/Restormer), especially the setup steps described in its `INSTALL.md`, to properly configure the project and install the required environment.

The training and testing commands for our EQ-Reg-based Restormer experiments are provided in the `README.md` files inside each task-specific subfolder. Please refer to the corresponding subdirectory for detailed usage instructions.

## 💌 Acknowledgement

* *This project is built using [**swinir**](https://github.com/JingyunLiang/SwinIR), [**KAIR**](https://github.com/cszn/KAIR), [**Restormer**](https://github.com/swz30/Restormer) repositories. We express our heartfelt gratitude for the contributions of these open-source projects.*
* *We also want to express our gratitude for some articles introducing this project and derivative implementations based on this project.*

