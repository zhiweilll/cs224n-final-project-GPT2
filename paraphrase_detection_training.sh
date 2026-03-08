#!/bin/bash
# 7.1 Cloze-Style Paraphrase Detection Training
# Trains ParaphraseGPT on the Quora dataset and generates prediction files.
#
# Output files:
#   predictions/para-dev-output.csv
#   predictions/para-test-output.csv
#
# Usage (on GCP VM with GPU):
#   bash paraphrase_detection_training.sh

set -e

python paraphrase_detection.py \
  --use_gpu \
  --model_size gpt2 \
  --epochs 10 \
  --lr 1e-5 \
  --batch_size 8 \
  --seed 11711 \
  --para_train data/quora-train.csv \
  --para_dev data/quora-dev.csv \
  --para_test data/quora-test-student.csv \
  --para_dev_out predictions/para-dev-output.csv \
  --para_test_out predictions/para-test-output.csv
