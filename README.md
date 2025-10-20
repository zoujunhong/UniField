
# UniField: Joint Multi-Domain Training for Universal Surface Pressure Modeling

<p align="center">
  <img src="images/architecture.png" alt="Overview of UniField" width="700">
</p>

> **UniField** is a universal flow-field modeling framework that learns shared aerodynamic representations across heterogeneous domains — including cars, trains, aircraft, and general shapes — via **joint multi-domain training**.  
> It integrates a *domain-agnostic geometric backbone* with *domain-specific Flow-Conditioned Adapters*, achieving cross-geometry and cross-velocity generalization.

[[Paper]](https://arxiv.org/abs/xxxx.xxxxx)

---

## 🧭 Overview

Traditional CFD simulations for aerodynamic analysis are computationally expensive and domain-specific.  
**UniField** addresses this limitation by **jointly training on multiple aerodynamic subfields**, enabling the model to learn *universal surface pressure representations* that transfer across different flow regimes.

Key features:
- **Unified Point Transformer backbone** for geometry encoding  
- **Parallel Flow-Conditioned Adapters (FCA)** for domain-specific flow conditioning  
- **Scalable architectures** (250M–2B parameters) for universal field representation  
- **State-of-the-art results** on [DrivAerNet++](https://github.com/Mohamedelrefaie/DrivAerNet) benchmark  

---

## 🚀 Usage

### Environment

| Dependency | Version |
|-------------|----------|
| Python      | 3.12 |
| PyTorch     | 2.6.0 + cu124 |
| CUDA GPU    | NVIDIA H100|

You can create the environment as:
```bash
conda create -n unifield python=3.12
conda activate unifield
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
````

---

### Training

```bash
python train.py
```


---

### Testing

```bash
python test.py
```

Evaluation results (MSE, MAE, RelL2, RelL1) will be shown in the command line.

---

## 🧩 Checkpoints

| Model Scale   | Parameters |  Download                          |
| ------------- | ---------- |  --------------------------------- |
| UniField-250M | 250M       |  [Download](https://www.baidu.com) |
| UniField-1B   | 1B         |  [Download](https://www.baidu.com) |
| UniField-2B   | 2B         |  [Download](https://www.baidu.com) |


---

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
******
```

---

## 📬 Contact

For questions or collaborations, please contact:
**Junhong Zou** – [zoujunhong2022@ia.ac.cn](mailto:zoujunhong2022@ia.ac.cn)


