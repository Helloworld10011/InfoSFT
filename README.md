# **InfoSFT: Learn More and Forget Less with Information-Aware Token Weighting**
[![arXiv](https://img.shields.io/badge/arXiv-2602.08813-b31b1b.svg)](https://arxiv.org/abs/2605.14967)
*InfoSFT is a modification of the standard SFT loss that guarantess better generalization on the new task while forgetting less.*

## Abstract
Supervised fine-tuning (SFT) provides the standard approach for teaching LLMs new behaviors from offline expert demonstrations. However, standard SFT uniformly fits all samples -- including those with low likelihood under the base model -- which can disproportionately drive training updates toward overfitting specific samples rather than learning the target behavior. Moreover, adapting to these unlikely samples induces substantial policy shifts that degrade prior capabilities. Existing methods mitigate this by filtering, regenerating, or down-weighting low-likelihood data. In doing so, they often suppress precisely the novel behaviors the base model has yet to learn. 

We propose InfoSFT, a principled weighting scheme for the SFT objective that concentrates learning signals on maximally informative, medium-confidence tokens -- those neither overly familiar to the base model nor too unlikely to cause instability. Requiring only a one-line modification to the standard token-wise loss, InfoSFT demonstrably improves generalization over vanilla SFT and likelihood-weighted baselines across math, code, and chain-of-thought tasks with diverse model families, while better preserving pre-existing capabilities.



