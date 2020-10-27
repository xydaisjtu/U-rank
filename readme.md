#!/usr/bin/env bash


# 1st:train the click model
python main.py -train_stage=click -batch_size=2048 --hparams learning_rate=7e-4


# 2nd:train the ranker
python main.py -train_stage=ranker -batch_size=1024 --hparams learning_rate=5e-4,pair_each_query=40


# test
python main.py -train_stage=click -decode=True
python main.py -train_stage=ranker -decode=True

