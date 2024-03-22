# Tuning Language Models by Proxy
This repository contains a presentation, pseudocode and code demonstration of Large Language Models (LLMs) based on the paper "Tuning Language Models by Proxy" by Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A. Smith. Here is the [link](https://arxiv.org/abs/2401.08565) to the paper.
  
Presenter: Yitian(Ewan) Long & Yunfei Lyu
  
## Table of Contents
- [Overview](#overview)
    - [Introduction](#introduction)
    - [Approach](#approach)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Pseudocode](#pseudocode)
- [Code Demonstration](#code-demonstration)
- [Critical Analysis](#critical-analysis)
- [Conclusion & Discussion](#conclusion--discussion)
- [Additional Resources](#additional-resources)
- [Citation](#citation)
  
## Overview
### Introduction
While the increasingly general capabilities of large language models (LLMs) have led to their widespread use, they still benefit from additional finetuning on specific tasks to achieve better performance. However, the finetuning process is often expensive and time-consuming, or impossible when model weights are private (e.g., GPT-4; OpenAI, 2023).
  
In this paper, the authors propose a new approach, *proxy-tuning*, which is a lightweight, decoding-time algorithm. This method allows for the adjustment of a large, black-box language model (LM) by utilizing the predictions from a lightweight model that operates on top of the LLM. The authors demonstrate that proxy-tuning can achieve competitive performance with traditional finetuning methods, while being more efficient and privacy-preserving, and can be used to tune LLMs without accessing their weights.
  
### Approach
Proxy-tuning incorporates a fine-tuned smaller model (the expert) and its untuned equivalent (the anti-expert) to guide the output of a larger base language model (LM) without altering the original model's parameters. 
  
- **Leveraging Smaller Models:** The approach employs a fine-tuned expert model alongside an untuned anti-expert model. The expert encapsulates the specific enhancements or knowledge from fine-tuning, while the anti-expert provides a baseline reference. These models are significantly smaller, making them less resource-intensive to fine-tune and manage.

- **Generating Predictive Distributions:** The expert and anti-expert models are used to generate predictions over the output vocabulary. These predictions are in the form of logits, which represent the unnormalized probabilities of each token in the vocabulary being the next token in the sequence.

- **Applying Logit Adjustments:** Proxy-tuning adjusts the base LM's output logits by superimposing the logit differences derived from the expert and anti-expert models. This process realigns the base LM’s predictions, effectively steering them towards the expert's fine-tuned characteristics.

## Methodology

The goal is to adjust the outputs of a large language model (referred to as M) by using two smaller models:

- An expert model (M+): This model has been fine-tuned to be good at a certain task.
- An anti-expert model (M−): This model has been fine-tuned to produce the opposite of the desired outcome.
Both M+ and M− are smaller and easier to fine-tune than the large model M.

Here's how it works:

- When  give an input xt(like a sentence or question) to the model,  also pass it to both M+ and M−.
- M+ and M− process the input and produce a set of scores for all possible outputs (like words or phrases that could come next).
- M+’s scores are added to M’s original scores, and - M−’s scores are subtracted from them. This is like saying, “Give me more of what M+ suggests and less of what M− doesn’t want.”
- After adjusting M’s scores with the scores from M+ and M−, the model uses a function called "softmax" to turn these scores into probabilities, which determines the likelihood of each possible output being the correct one.

![method section formula](figures/figure_2.png)

This method effectively "steers" the large model to produce outputs more like what we want (based on M+) and avoid what we don’t want (based on M−), without having to go through the costly process of fine-tuning M directly.

![Proxy-tuning adjusts a large pretrained model's predictions using the logit differences from a fine-tuned "expert" and an untuned "anti-expert," without changing the model's internal weights.](figures/figure_1.png "Proxy-Tuning: Steering Pretrained Models with Expert Logit Differences")


