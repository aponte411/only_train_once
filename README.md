# Only Train Once (OTO): Automatic One-Shot DNN Training And Compression Framework

[![OTO-bage](https://img.shields.io/badge/OTO-red?logo=atom&logoColor=white)](#) [![autoML-bage](https://img.shields.io/badge/autoML-blue?logo=dependabot&logoColor=white)](#) [![DNN-training-bage](https://img.shields.io/badge/DNN-training-yellow)](#) [![DNN-compress-bage](https://img.shields.io/badge/DNN-compress-purple)](#) [![Operator-pruning-bage](https://img.shields.io/badge/Operator-pruning-green)](#) [![Operator-erasing-bage](https://img.shields.io/badge/Operator-erasing-CornflowerBlue)](#) [![build-pytorchs-bage](https://img.shields.io/badge/build-pytorch-orange)](#) [![lincese-bage](https://img.shields.io/badge/license-MIT-blue.svg)](#) [![prs-bage](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#)

### Please note.

Merry Christmas. Our next-version OTO library is being released. The new library has brand new designs along with a lot of major improvements. Please be patient for the release to be accompolished. The README will be updated accordingly in the coming days. 

The previous repo has been moved into [legacy_branch](https://github.com/tianyic/only_train_once/tree/otov2_legacy_backup) for academic replication.

![oto_overview](https://github.com/tianyic/only_train_once/assets/8930611/131bd6ba-3f94-4b46-8398-074ae311ccf0)

This repository is the Pytorch implementation of **Only-Train-Once** (**OTO**). OTO is an $\color{LimeGreen}{\textbf{automatic}}$, $\color{LightCoral}{\textbf{architecture}}$ $\color{LightCoral}{\textbf{agnostic}}$ DNN $\color{Orange}{\textbf{training}}$ and $\color{Violet}{\textbf{compression}}$ (via $\color{CornflowerBlue}{\textbf{structure pruning}}$ and $\color{DarkGoldenRod}{\textbf{erasing}}$ operators) framework. By OTO, users could train a general DNN either from scratch or a pretrained checkpoint to achieve both high performance and slimmer architecture simultaneously in the one-shot manner (without fine-tuning). 

## Publications

Please find our series of works and [bibtexs](https://github.com/tianyic/only_train_once?tab=readme-ov-file#citation) for kind citations. 

- [OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators](https://arxiv.org/abs/2312.09411) preprint.
- [LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery](https://arxiv.org/abs/2310.18356) preprint. 
- [An Adaptive Half-Space Projection Method for Stochastic Optimization Problems with Group Sparse Regularization](https://openreview.net/pdf?id=KBhSyBBeeO) in TMLR 2023.  
- [OTOv2: Automatic, Generic, User-Friendly](https://openreview.net/pdf?id=7ynoX1ojPMt) in ICLR 2023.
- [Only Train Once (OTO): A One-Shot Neural Network Training And Pruning Framework](https://papers.nips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html) in NeurIPS 2021.

![oto_overview_2](https://github.com/tianyic/only_train_once/assets/8930611/ed1f8fda-d43c-4b60-a627-7ce9b2277848)

## Installation

We recommend to run the framework under `pytorch>=2.0`. Use `pip` or `git clone` to install.

```bash
pip install only_train_once
```
or
```bash
git clone https://github.com/tianyic/only_train_once.git
```

## Quick Start

We provide an example of OTO framework usage. More explained details can be found in [tutorals](./tutorials/).

### Minimal usage example. 

```python
import torch
from sanity_check.backends import densenet121
from only_train_once import OTO

# Create OTO instance
model = densenet121()
dummy_input = torch.zeros(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# Create HESSO optimizer
optimizer = oto.hesso(variant='sgd', lr=0.1, target_group_sparsity=0.7)

# Train the DNN as normal via HESSO
model.train()
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(max_epoch):
    f_avg_val = 0.0
    for X, y in trainloader:
        X, y = X.cuda(), y.cuda()
        y_pred = model.forward(X)
        f = criterion(y_pred, y)
        optimizer.zero_grad()
        f.backward()
        optimizer.step()

# A compressed densenet will be generated. 
oto.construct_subnet(out_dir='./')
```

## How the pruning mode in OTO works.

- **Pruning Zero-Invariant Group Partition.** OTO at first automatically figures out the dependancy inside the target DNN to build a pruning dependency graph. Then OTO partitions DNN's trainable variables into so-called Pruning Zero-Invariant Groups (PZIGs). PZIG describes a class of pruning minimally removal structure of DNN, or can be largely interpreted as the minimal group of variables that must be pruned together. 
![zig_partition](https://user-images.githubusercontent.com/8930611/224582957-d3955a50-2abc-44b7-b134-1ba0075ca85f.gif)


- **Hybrid Structured Sparse Optimizer.** A structured sparsity optimization problem is formulated. A hybrid structured sparse optimizer, including HESSO, DHSPG, LSHPG, is then employed to find out which PZIGs are redundant, and which PZIGs are important for the model prediction. The selected hybrid optimizer explores group sparsity more reliably and typically achieves higher generalization performance than other sparse optimizers.
![dhspg](https://user-images.githubusercontent.com/8930611/224577550-3814f6c9-0eaf-4d1c-a978-2251b68c2a1a.png)


- **Construct pruned model.** The structures corresponding to redundant PZIGs (being zero) are removed to form the pruned model. Due to the property of PZIGs, **the pruned model returns the exact same output as the full model**. Therefore, **no further fine-tuning** is required. 
<p align="center"><img width="400" alt="comp_construct" src="https://user-images.githubusercontent.com/8930611/224575936-27594b36-1d1d-4daa-9f07-d125dd6e195e.png"></p> 

## More full and compressed models

Please find more full and compressed models by OTO on [checkpoints](https://drive.google.com/drive/folders/1lZ7Wsehi0hr_g8nztbAFEJIhF8C4Q8Kp?usp=share_link). The full and compressed models return the exact same outputs given the same inputs.

The dependancy graphs for ZIG partition can be found at [Dependancy Graphs](https://drive.google.com/drive/folders/1XVRUEr4cUyT6xVknLF2SsYKgXBZ0gjeD?usp=share_link).

## Remarks and to do list

The current OTO library depends on 

- The target model needed to be convertable into ONNX format for conducting dependancy graph construction.

- Please check our supported [operators](./only_train_once/operation/operators_dict.py) list if meeting some errors.

- The effectiveness (ultimate compression ratio and model performance) relies on the proper usage of DHSPG optimizer. Please go through our [tutorials](./tutorials/) for setup (will be kept updated).

We will routinely complete the following items.

- Provide more tutorials to cover more use cases and applications of OTO. 

- Provide documentations of the OTO API.

- Optimize the dependancy list.

## Welcome Contributions

We greatly appreciate the contributions from our open-source community to make DNN's training and compression to be more automatic and convinient. 

## Citation

If you find the repo useful, please kindly star this repository and cite our papers:

```bibtex
For OTOv3 preprint
@article{chen2023otov3,
  title={OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators},
  author={Chen, Tianyi and Ding, Tianyu and Zhu, Zhihui and Chen, Zeyu and Wu, HsiangTao and Zharkov, Ilya and Liang, Luming},
  journal={arXiv preprint arXiv:2312.09411},
  year={2023}
}

For LoRAShear preprint
@article{chen2023lorashear,
  title={LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery},
  author={Chen, Tianyi and Ding, Tianyu and Yadav, Badal and Zharkov, Ilya and Liang, Luming},
  journal={arXiv preprint arXiv:2310.18356},
  year={2023}
}

For AdaHSPG+ publication in TMLR (theoretical optimization paper)
@article{dai2023adahspg,
  title={An adaptive half-space projection method for stochastic optimization problems with group sparse regularization},
  author={Dai, Yutong and Chen, Tianyi and Wang, Guanyi and Robinson, Daniel P},
  journal={Transactions on machine learning research},
  year={2023}
}

For OTOv2 publication in ICLR 2023
@inproceedings{chen2023otov2,
  title={OTOv2: Automatic, Generic, User-Friendly},
  author={Chen, Tianyi and Liang, Luming and Tianyu, DING and Zhu, Zhihui and Zharkov, Ilya},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

For OTOv1 publication in NeurIPS 2021
@inproceedings{chen2021otov1,
  title={Only Train Once: A One-Shot Neural Network Training And Pruning Framework},
  author={Chen, Tianyi and Ji, Bo and Tianyu, DING and Fang, Biyi and Wang, Guanyi and Zhu, Zhihui and Liang, Luming and Shi, Yixin and Yi, Sheng and Tu, Xiao},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
