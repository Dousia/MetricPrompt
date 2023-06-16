#!/bin/bash


python -W ignore ../main.py \
  --dataset $1 \
  --k_shot $2 \
  --n_adapt_epochs $3 \
  --prompt_template 0 \
  --pivot 0 \
  --start_episode 0 \
  --num_episodes 10 \
  --kernl_accerleration 0 \
  --seed 1999


