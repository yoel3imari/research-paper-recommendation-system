# Evaluation

Proxies used (no citation graph in this dataset — architecture §7): category-purity@k vs a random baseline, plus a hand-judged query set.

## Category-purity@k

Seed papers sampled: 200. Purity above the random
baseline indicates the embeddings capture topical structure.

| k | Purity@k | Random baseline | Lift |
|---|---|---|---|
| 1 | 1.0000 | 0.2298 | +0.7702 |
| 5 | 0.7070 | 0.2298 | +0.4772 |
| 10 | 0.6500 | 0.2298 | +0.4202 |

## Hand-judged queries (relevance 0–3)

### retrieval augmented generation for code

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1709.07642` | Code Attention: Translating Code to Comments by Exploiting Domain
  Features | 0.538 |  |
| 2 | `1703.05698` | Neural Sketch Learning for Conditional Program Generation | 0.527 |  |
| 3 | `1409.7165` | Heterogeneous Metric Learning with Content-based Regularization for
  Software A | 0.523 |  |
| 4 | `1608.09000` | Learning Syntactic Program Transformations from Examples | 0.508 |  |
| 5 | `1210.2195` | Annotating Answer-Set Programs in LANA? | 0.508 |  |

### efficient transformer inference

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1611.08230` | Learning Fast Sparsifying Transforms | 0.476 |  |
| 2 | `1103.0598` | Learning transformed product distributions | 0.460 |  |
| 3 | `1502.03509` | MADE: Masked Autoencoder for Distribution Estimation | 0.458 |  |
| 4 | `1311.4158` | Unsupervised Learning of Invariant Representations in Hierarchical
  Architectur | 0.448 |  |
| 5 | `1705.08881` | Dense Transformer Networks | 0.447 |  |

### graph neural networks for molecular property prediction

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1706.09916` | Graph Convolution: A High-Order and Adaptive Approach | 0.663 |  |
| 2 | `1709.03741` | Learning Graph-Level Representation for Drug Discovery | 0.662 |  |
| 3 | `1704.01212` | Neural Message Passing for Quantum Chemistry | 0.641 |  |
| 4 | `1711.07553` | Residual Gated Graph ConvNets | 0.603 |  |
| 5 | `1711.05859` | Hybrid Approach of Relation Network and Localized Graph Convolutional
  Filterin | 0.599 |  |

### reinforcement learning from human feedback

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1606.06979` | Simultaneous Control and Human Feedback in the Training of a Robotic
  Agent wit | 0.607 |  |
| 2 | `1703.01274` | Actor-Critic Reinforcement Learning with Simultaneous Human Control and
  Feedba | 0.587 |  |
| 3 | `1604.03912` | Inverse Reinforcement Learning with Simultaneous Estimation of Rewards
  and Dyn | 0.557 |  |
| 4 | `1701.02369` | Reinforcement Learning based Embodied Agents Modelling Human Users
  Through Int | 0.537 |  |
| 5 | `1202.1558` | On the Performance of Maximum Likelihood Inverse Reinforcement Learning | 0.531 |  |

### vision transformers for medical imaging

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1401.0166` | Medical Image Fusion: A survey of the state of the art | 0.494 |  |
| 2 | `1501.02859` | $\ell_0$ Sparsifying Transform Learning with Efficient Optimal Updates
  and Con | 0.447 |  |
| 3 | `1511.01289` | Data-Driven Learning of a Union of Sparsifying Transforms Model for
  Blind Comp | 0.423 |  |
| 4 | `1611.02443` | Domain Adaptation with L2 constraints for classifying images from
  different en | 0.416 |  |
| 5 | `1406.2639` | A New 2.5D Representation for Lymph Node Detection using Random Sets of
  Deep C | 0.413 |  |

### diffusion models for text generation

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1601.07969` | Zipf's law is a consequence of coherent language production | 0.583 |  |
| 2 | `0710.5382` | Some Reflections on the Task of Content Determination in the Context of
  Multi- | 0.557 |  |
| 3 | `1203.3463` | Timeline: A Dynamic Hierarchical Dirichlet Process Model for Recovering
  Birth/ | 0.553 |  |
| 4 | `1703.09902` | Survey of the State of the Art in Natural Language Generation: Core
  tasks, app | 0.542 |  |
| 5 | `1611.09235` | Joint Copying and Restricted Generation for Paraphrase | 0.539 |  |

### federated learning privacy

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1502.06309` | Learning with Differential Privacy: Stability, Learnability and the
  Sufficienc | 0.655 |  |
| 2 | `1610.05755` | Semi-supervised Knowledge Transfer for Deep Learning from Private
  Training Dat | 0.644 |  |
| 3 | `1702.07464` | Deep Models Under the GAN: Information Leakage from Collaborative Deep
  Learnin | 0.629 |  |
| 4 | `1705.08197` | Learning to Succeed while Teaching to Fail: Privacy in Closed Machine
  Learning | 0.622 |  |
| 5 | `1703.00403` | Preserving Differential Privacy Between Features in Distributed
  Estimation | 0.618 |  |

### large language model evaluation benchmarks

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1312.3005` | One Billion Word Benchmark for Measuring Progress in Statistical
  Language Mode | 0.668 |  |
| 2 | `1605.04515` | Meta-Evaluation of Translation Evaluation Methods: a systematic up-to-date overv | 0.645 |  |
| 3 | `1703.08748` | LEPOR: An Augmented Machine Translation Evaluation Metric | 0.632 |  |
| 4 | `1510.04600` | Telemedicine as a special case of Machine Translation | 0.599 |  |
| 5 | `1710.08691` | BENGAL: An Automatic Benchmark Generator for Entity Recognition and
  Linking | 0.597 |  |

### neural architecture search

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1611.01578` | Neural Architecture Search with Reinforcement Learning | 0.638 |  |
| 2 | `1710.10776` | Transfer Learning to Learn with Multitask Neural Model Search | 0.629 |  |
| 3 | `1704.08792` | DeepArchitect: Automatically Designing and Training Deep Architectures | 0.619 |  |
| 4 | `1707.04873` | Efficient Architecture Search by Network Transformation | 0.619 |  |
| 5 | `1707.07012` | Learning Transferable Architectures for Scalable Image Recognition | 0.567 |  |

### multi-agent language models

| # | arXiv id | Title | Score | Relevance (0–3) |
|---|---|---|---|---|
| 1 | `1705.11192` | Emergence of Language with Multi-agent Games: Learning to Communicate
  with Seq | 0.614 |  |
| 2 | `1602.03661` | On the emergence of syntactic structures: quantifying and modelling
  duality of | 0.605 |  |
| 3 | `1111.0041` | On the Formal Semantics of Speech-Act Based Communication in an
  Agent-Oriented | 0.605 |  |
| 4 | `1706.08502` | Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog | 0.600 |  |
| 5 | `1612.07182` | Multi-Agent Cooperation and the Emergence of (Natural) Language | 0.572 |  |
