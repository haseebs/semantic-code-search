# Semantic Code Search

This repo contains the unofficial pytorch implementation of some of the experiments in ["CodeSearchNet Challenge: Evaluating the State of Semantic Code Search"](https://arxiv.org/abs/1909.09436). The original tensorflow implementation can be found [here](https://github.com/github/CodeSearchNet).

## A few instructions

- Set up the requirements using `requirements.txt`.
- Download and set up the dataset using the links provided in the [original implementation](https://github.com/github/CodeSearchNet).

## W&B Setup

You need to initialize W&B to log the results. If it's your first time using W&B on a machine, you will need to log in:
```
$ wandb login
```
You will be asked for your API key, which appears on your W&B profile settings page.

