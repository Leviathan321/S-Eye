#!/bin/bash

python -m maxitwo.train_model \
    --env-name "udacity" \
    --archive-names "udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz" \
    --model-name "dave2_MCD" \
    --seed 0 \
    --archive-path "C:\Users\sorokin\Downloads\training_datasets\\" \
    --model-save-path "C:\Users\sorokin\Documents\Projects\uq-testing\ruben-misbehaviour-prediction\models\MAXITWO\MCD" \
    --nb-epoch 50 \
    --use-dropout True \
    --drop-rate-start 0.05 \
    --drop-rate-end 0.05