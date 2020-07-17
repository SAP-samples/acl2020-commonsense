
# Contrastive Self-Supervised Learning for Commonsense Reasoning
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

#### News
- 07/17/2020: Provided source code
- 05/15/2020: Provided model for download
 
This repository contains the source code for our paper [Contrastive Self-Supervised Learning for Commonsense Reasoning](https://arxiv.org/abs/2005.00669) to be presented at  [ACL2020]( https://acl2020.org/). The code is in parts based on the code from the paper [A Surprisingly Robust Trick for Winograd Schema Challenge](https://github.com/vid-koci/bert-commonsense).


 
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


## Download and Installation

1. Clone this repository and install dependencies:
```
git clone https://github.com/SAP/acl2020-commonsense-reasoning
cd acl2020-commonsense-reasoning
pip install -r requirements.txt
```

## Model

The BERT-Large model is available in the Huggingface repository as [sap-ai-research/BERT-Large-Contrastive-Self-Supervised-ACL2020](https://huggingface.co/sap-ai-research/BERT-Large-Contrastive-Self-Supervised-ACL2020).

Loading the model in Python:

```
tokenizer = AutoTokenizer.from_pretrained("sap-ai-research/BERT-Large-Contrastive-Self-Supervised-ACL2020")

model = AutoModelWithLMHead.from_pretrained("sap-ai-research/BERT-Large-Contrastive-Self-Supervised-ACL2020")
```

This model should reproduce the results reported in the paper:

```
Knowref-test:  0.6558966074313409
DPR/WSCR-test:  0.8014184397163121
WSC:  0.6959706959706959
PDP:  0.9
```

## Related work
See our work accepted [ACL'19](http://acl2019.org/) - *Attention Is (not) All You Need for Commonsense Reasoning* - proposing BERT attention-guidance for commonsense reasoning. [arXiv](https://arxiv.org/abs/1905.13497), [GitHub](https://github.com/SAP-samples/acl2019-commonsense/)

## Known Issues
No issues known


## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.


## Citations
If you use this code in your research,
please cite:

```
@inproceedings{klein-nabi-2020-contrastive,
    title = "Contrastive Self-Supervised Learning for Commonsense Reasoning",
    author = "Klein, Tassilo  and
      Nabi, Moin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.671",
    pages = "7517--7523"
}
```


## License
Copyright (c) 2019 SAP SE or an SAP affiliate company. All rights reserved. This file is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE file](LICENSE).
