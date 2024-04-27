# VLPT -  Video and Langauge Pre-Training
Embodying Language Models via PaLI/Flamingo-style fusion and VPT-style pretraining

This was an ambitious ML research & engineering project for my year 3 dissertation in the University of Bath.

The project started in October 2022, looking at the new multimodal visual language models Flamingo (https://arxiv.org/abs/2204.14198) and PaLI (https://arxiv.org/abs/2209.06794) as well as paper "Video Pre-Training" (https://arxiv.org/abs/2206.11795), which provided a method to transform unlabelled video data into a pseudo-labelled behavioural cloning dataset. This work sought to combine these two ideas for the purpose of embodying LLMs into an agent which works at the level of raw video pixels, mouse movements and keystrokes.

# Approach
This involved much data engineering. We use some methods from the VPT paper to automatically scrape and clean 800 hours of Minecraft video data, but end up mostly manually searching due to unsatisfactory accuracy from the filter and realising that, at the scale we were capable of training at, the data required was small enough and quality crucial enough that this was feasible and likely important. It also required filtering for videos with a consistent language signal (i.e. we use only English videos within a range of WPMs) and transcribing and timestamping them at the token-level.

Much deliberation went into efficient ways to implement a multimodal cross-attention mechanism such that it was efficient, maintains causality, keeps the video and audio signals aligned, and works autoregressively. This required much looking into the models and repos for VPT (https://github.com/openai/Video-Pre-Training), TransformerXL (https://github.com/kimiyoung/transformer-xl), and Flamingo (https://github.com/lucidrains/flamingo-pytorch), especially the VPT codebase (https://github.com/openai/Video-Pre-Training), which this code builds on top of.

This repository provides all the code I used, from the various data gathering stages (including discarded methods) to training.

# Results
As the model trains, we can see the language modelling and action modelling losses decrease, as well as an increasing dependence between the two models (we use gated cross attention as in the Flamingo paper, and as in that paper we see the gate values increase across training). Unfortunately, due to issues with the VPT repo (we couldn't figure out how to use even the unmodified VPT repo to finetune an agent without it becoming erratic when rolled out into the environment), time constraints and (potentially) VRAM constraints (its unclear if increasing batch size would have solved instabilities), the resulting agent behaves erratically and we're unable to confirm it has learnt a relation between language and actions.

However, I am proud of what I accomplished and have gained a lot of experience with AI/ML research at all stages, including the identification of interesting research questions; constructing efficient data engineering pipelines; I am now very confident in my undertanding of transformers and some of their variants and in training them; how these kinds of huge ML projects like VPT are laid out and done in reality; using compute budgets wisely; and finding key stats to evaluate novel architectures.

I would like to get back to this, especially now that there are other works we can reference which build from VPT and are able to finetune the agent successfuly, but my research interests have changed more towards interpretability, thinking about the conundrum between parallelism, complexity and state-tracking (https://blog.wtf.sg/posts/2023-02-03-the-new-xor-problem/ https://arxiv.org/abs/2404.08819), and investigating OOD generalization in ICL & SGD, so this remains on the to-do list.

# Key Files

VLPT.pdf contains the dissertation itself. The last appendix of this describes all the important files in this repository.

main.ipynb illustrates how all the different files were executed to carry out the dissertation, neatly explaining the order in which it's all run.
