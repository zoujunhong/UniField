# Paper Information

This repository contains implementations related to AdaField and UniField.

## AdaField

- **Title:** AdaField: Generalizable Surface Pressure Modeling with Physics-Informed Pre-training and Flow-Conditioned Adaptation
- **Status:** AAAI 2026
- **Link:** https://arxiv.org/abs/2601.07139
- **Authors:** Junhong Zou, Wei Qiu, Zhenxu Sun, Xiaomei Zhang, Zhaoxiang Zhang, Xiangyu Zhu

**Abstract summary:** AdaField targets surface pressure prediction for transportation geometries under limited-data conditions. It combines a Semantic Aggregation Point Transformer backbone with flow-conditioned adapters and physics-informed data augmentation, aiming to transfer from large public aerodynamic datasets to data-scarce domains such as high-speed trains and aircraft.

## UniField

- **Title:** UniField: Joint Multi-Domain Training for Universal Surface Pressure Modeling
- **Status:** arXiv preprint
- **Link:** https://arxiv.org/abs/2510.24106
- **Authors:** Junhong Zou, Zhenxu Sun, Yueqing Wang, Wei Qiu, Zhaoxiang Zhang, Xiangyu Zhu, Zhen Lei

**Abstract summary:** UniField studies joint multi-domain training for aerodynamic surface pressure prediction across automobiles, trains, aircraft, and general geometries. It uses a shared geometry encoder with domain-specific flow-conditioned normalization, and introduces broader geometry/flow diversity through ThingiCFD to improve cross-domain generalization and performance in data-scarce settings.
