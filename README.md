# VLPT -  Video and Langauge Pre-Training
Embodying Language Models via PaLI-style fusion and VPT-style pretraining

This was an ambitious engineering & ML research project for my year 3 dissertation.

The idea was basically looking at the new (at the time) multimodal VLMs like Flamingo (https://arxiv.org/abs/2204.14198) and PaLI (https://arxiv.org/abs/2209.06794); and the new ability to tap unlabelled video data for behavioural cloning datasets from Video Pre-Training (https://arxiv.org/abs/2206.11795); and combining the two for the purpose of embodying LLMs into an agent which works at the level of raw pixels, mouse movements and keystrokes.

This involved much data gathering, much deliberation about efficient ways to implement multimodal cross attention in a video-friendly, autoregressive way, and much working into understanding the models and repos for VPT (https://github.com/openai/Video-Pre-Training), TransformerXL (https://github.com/kimiyoung/transformer-xl), and Flamingo (https://github.com/lucidrains/flamingo-pytorch) which this work starts from (especially the VPT codebase, since we re-use the training loop from there and this makes up the majority of the complexity and code of the model).

This repository provides all the code I used, from the various data gathering stages (including discarded methods) to training.

As the model trains we can see language modelling and action mdelling losses decrease, as well as increasing dependence between the two models (we use gated cross attention as in the Flamingo apper, and like that paper we see the gate values increase across training). Unfortunately, due to issues with the VPT repo (we couldn't figure out how to use even the unmodified VPT repo to finetune an agent without it becoming erratic), time constraints and (potentially) VRAM constraints (its unclear if increasing batch size would have solved instabilities), the resulting agent behaves erratically, its language degenerates a few seconds into the episode, and we're unable to reach conclusive results on the learnt relation between language and actions.

I would like to get back to this, especially now that there are other works we can reference which build from VPT and are able to finetune the agent successfuly, but my research interests have changed more towards interpretability, scalable adaptive compute time & recurrence, and investigating ICL & SGD, so this remains on the to-do list.

# Key Files

VLPT.pdf contains the dissertation itself. The last appendices of this describe all the important files and code, as well as the file structure of the project.

main.ipynb illustrates how all the different files were executed to carry out the dissertation, neatly explaining the order in which it's all run.
