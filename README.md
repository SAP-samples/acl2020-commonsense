
# Contrastive Self-Supervised Learning for Commonsense Reasoning
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Note: Source code will be provided soon.
 
This repository will contain the source code for our paper [Contrastive Self-Supervised Learning for Commonsense Reasoning](https://arxiv.org/abs/2005.00669) to be presented at  [ACL2020]( https://acl2020.org/).


 
We propose a self-supervised method to solve *Pronoun Disambiguation* and *Winograd Schema Challenge* problems.
Our approach exploits the characteristic structure of training corpora related to so-called *trigger* words, which are responsible for flipping the answer in pronoun disambiguation. 
We achieve such commonsense reasoning by constructing pair-wise contrastive auxiliary predictions. To this end, we leverage a *mutual exclusive loss* regularized by a *contrastive* margin.
Our architecture is based on the recently introduced transformer networks, BERT, that exhibits strong performance on many language understanding benchmarks. Empirical results show that our method alleviates the limitation of current supervised approaches for commonsense reasoning. This study opens up avenues for exploiting inexpensive self-supervision to achieve performance gain in commonsense reasoning tasks.

#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)

## Requirements
- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Huggingface Tranformers](https://github.com/huggingface/transformers)

## Known Issues
No issues known


## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.


## Citations
If you use this code in your research,
please cite:

```
@misc{klein2020contrastive,
    title={Contrastive Self-Supervised Learning for Commonsense Reasoning},
    author={Tassilo Klein and Moin Nabi},
    year={2020},
    eprint={2005.00669},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


## License
Copyright (c) 2019 SAP SE or an SAP affiliate company. All rights reserved. This file is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE file](LICENSE).
