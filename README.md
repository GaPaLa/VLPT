# VLPT -  Video and Langauge Pre-Training
Embodying Language Models via PaLI/Flamingo-style fusion and VPT-style pretraining

This was an ambitious engineering & ML research project for my year 3 dissertation.

The project started in October 2022, looking at the new multimodal VLMs (Flamingo (https://arxiv.org/abs/2204.14198) and PaLI (https://arxiv.org/abs/2209.06794)) as well as a new method to tap unlabelled video data for behavioural cloning datasets provided by the "Video Pre-Training" paper (https://arxiv.org/abs/2206.11795); and sought to combine the two for the purpose of embodying LLMs into an agent which works at the level of raw video pixels, mouse movements and keystrokes.

This involved much data gathering. We use some methods from the VPT paper to automatically scrape and clean 800 hours of Minecraft video data, but end up mostly manually searching due to poor accuracy from the filter and realising that the target scale was small enough that this was feasible and that dataset quality would be more important at this scale. It also required filtering for videos with a clean language signal (i.e. we use only English videos within a WPM range) and transcribing and timestamping them at the token-level.

Much deliberation went into efficient ways to implement a multimodal cross-attention mechanism such that it was efficient, maintains causality, keeps the video and audio signals aligned, and works autoregressively. This required much looking into the models and repos for VPT (https://github.com/openai/Video-Pre-Training), TransformerXL (https://github.com/kimiyoung/transformer-xl), and Flamingo (https://github.com/lucidrains/flamingo-pytorch), especially the VPT codebase (https://github.com/openai/Video-Pre-Training), which this code builds from.

This repository provides all the code I used, from the various data gathering stages (including discarded methods) to training.

As the model trains, we can see the language modelling and action mdelling losses decrease, as well as an increasing dependence between the two models (we use gated cross attention as in the Flamingo paper, and like in that paper we see the gate values increase across training). Unfortunately, due to issues with the VPT repo (we couldn't figure out how to use even the unmodified VPT repo to finetune an agent without it becoming erratic), time constraints and (potentially) VRAM constraints (its unclear if increasing batch size would have solved instabilities), the resulting agent behaves erratically and we're unable to confirm it has learnt a relation between language and actions.

However, I did learn a lot about data collection pipelines and automating them; I am now very confident in my undertanding of every detail of transformers, their derivatives and training them; how these kinds of huge ML projects like VPT are laid out and done in reality; using compute budgets wisely; and a lot more which was both fun and will definitely come in handy!

I would like to get back to this, especially now that there are other works we can reference which build from VPT and are able to finetune the agent successfuly, but my research interests have changed more towards interpretability, scalable adaptive compute time & recurrence, and investigating ICL & SGD, so this remains on the to-do list.

# Key Files

VLPT.pdf contains the dissertation itself. The last appendix of this describe all the important files and code in this repository and its structure.

main.ipynb illustrates how all the different files were executed to carry out the dissertation, neatly explaining the order in which it's all run.
