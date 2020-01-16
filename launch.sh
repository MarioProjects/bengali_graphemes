#!/bin/bash

model="densenet121"  # densenet121 - seresnext101 - efficientnetb7b
head="initial_head"  # initial_head - initial_head_noDrop
#loss="reduction_loss"  # standard_loss - reduction_loss
criterion="combined_crossentropy"  #combined_crossentropy
#optimizer="adam"
data_augmentation="da7"
lr=0.001 # learning_rate
epochs=29
batch_size=128
img_size=150
crop_size=128
validation_size=0.15
additional_info=''
mixup=0.0

CUDA_VISIBLE_DEVICES=0,1 python3 -u train.py --epochs $epochs --batch_size $batch_size \
                        --learning_rate $lr  --img_size $img_size --crop_size $crop_size --criterion $criterion \
                        --slack_resume --data_augmentation $data_augmentation --mixup_alpha $mixup \
                        --model_name $model --head_name $head --validation_size $validation_size
#                        --additional_info $additional_info
