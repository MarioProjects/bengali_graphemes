#!/bin/bash

model="seresnext101"  # densenet121 - seresnext101 - efficientnetb7b
head="initial_head"  # initial_head - initial_head_noDrop
#loss="reduction_loss"  # standard_loss - reduction_loss
criterion="combined_crossentropy"  #combined_crossentropy
optimizer="sgd"
scheduler="steps"
scheduler_step=150
scheduler_decay=0.1
data_augmentation="da7b"

epochs=96
batch_size=50
img_size=224
crop_size=224
validation_size=0.1
additional_info=''

mixup_prob=0.0
mixup_alpha=0.4
cutmix_prob=0.0
cutmix_alpha=1.0

grad_clipping=1.0  # No clip -> 9999

model_checkpoint="best_model.pth"

for min_lr in 0.0457
do

max_lr=$min_lr
lr=$min_lr # learning_rate

CUDA_VISIBLE_DEVICES=0,1 python3 -u train.py --epochs $epochs --batch_size $batch_size \
    --img_size $img_size --crop_size $crop_size --criterion $criterion --pretrained \
    --slack_resume --data_augmentation $data_augmentation --grad_clipping $grad_clipping \
    --model_name $model --head_name $head --validation_size $validation_size \
    --min_lr $min_lr --max_lr $max_lr --learning_rate $lr --optimizer $optimizer \
    --mixup_prob $mixup_prob --mixup_alpha $mixup_alpha --cutmix_prob $cutmix_prob --cutmix_alpha $cutmix_alpha \
    --scheduler $scheduler --scheduler_step $scheduler_step --scheduler_decay $scheduler_decay \
    --model_checkpoint $model_checkpoint --apply_swa
    #--additional_info $additional_info

done