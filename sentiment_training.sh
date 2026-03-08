#!/bin/bash
# 6.3 Training GPT-2 for Sentiment Classification

set -e

# sentiment classification (full model)
python classifier.py --fine-tune-mode full-model --use_gpu --batch_size 64 --lr 1e-5 --hidden_dropout_prob 0.1 --epochs 10

# sentiment classification (last linear layer)
python classifier.py --fine-tune-mode last-linear-layer --use_gpu --batch_size 64 --lr 1e-3 --hidden_dropout_prob 0.1 --epochs 10
