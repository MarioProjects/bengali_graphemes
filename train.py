#!/usr/bin/env python
# coding: utf-8

# This kernel is based on [Heng's Starter code](https://www.kaggle.com/c/bengaliai-cv19/discussion/123757).
# I have already published a [kernel](https://www.kaggle.com/bibek777/heng-starter-inference-kernel)
# doing the inference using his models. In this kernel, we will use his codes to do the training.

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.


# Add heng's code to our envionment and import the modules


from common import *
from file import *
from models import *
from utils.arguments.train_arguments import *
from utils.training import *
from utils.data import *
from utils.logging_functions import *
from kaggle import *
from os import environ

# Heng uses his own version of datasplit(which I have uploaded [here](https://www.kaggle.com/bibek777/hengdata)).
# I tried using it but get memory error, maybe it's too large to load. So I have edited the dataloader in his code
# and use different split for train and valid dataset. The codes/ideas for dataloader is taken from this
# [kernel](https://www.kaggle.com/backaggle/catalyst-baseline). Also the dataset used in this kernel is taken
# from [here](https://www.kaggle.com/pestipeti/bengaliai), uploaded by Peter

if environ.get('BENGALI_DATA_PATH') is not None:
    INPUT_PATH = environ.get('BENGALI_DATA_PATH')
    TRAIN_IMGS = "grapheme-imgs"
    LABELS = INPUT_PATH + "/train.csv"
else:
    assert False, "Please set the environment variable BENGALI_DATA_PATH. Read the README!"

data_root = "{}/bengaliai/256_train/256/".format(INPUT_PATH)
df = pd.read_csv("{}/train.csv".format(INPUT_PATH))
train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=2019)

TASK_NAME = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
NUM_TASK = len(TASK_NAME)

img_size, crop_size = 256, 224
train_augment_albumentation, valid_augment_albumentation = data_augmentation_selector(args.data_augmentation,
                                                                                      args.img_size, args.crop_size)

def run_train():
    out_dir = 'kaggle/working'
    initial_checkpoint = None

    scheduler = DecayScheduler(base_lr=args.learning_rate, decay=0.1, step=50)
    iter_accum = 1

    for f in ['checkpoint', 'train', 'valid']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)

    train_dataset = BengaliDataset(df=train_df, data_path=data_root, augment=train_augment_albumentation)
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size,
                              drop_last=True, num_workers=0, pin_memory=True, collate_fn=null_collate
                              )

    valid_dataset = BengaliDataset(df=valid_df, data_path=data_root, augment=valid_augment_albumentation)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=args.batch_size,
                              drop_last=False, num_workers=0, pin_memory=True, collate_fn=null_collate
                              )

    net = model_selector(args.model_name, args.head_name, [168, 11, 7], pretrained=args.pretrained).cuda()

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=scheduler(0))
    # optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=scheduler(0), momentum=0.0, weight_decay=0.0)

    criterion = select_criterion(args.criterion)

    if args.model_checkpoint != "":
        print("Loading model from checkpoint...")
        # Because we used multiple GPUs training
        state_dict = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
        from collections import OrderedDict

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict, strict=False)


    # # num_iters   = 3000*1000 # use this for training longer
    # num_iters = 1000  # comment this for training longer
    # iter_smooth = 50
    # iter_log = 250
    # iter_valid = 500

    num_iters = int(len(train_dataset) / args.batch_size) * args.epochs
    print(int(len(train_dataset) / args.batch_size))
    iter_smooth = 50
    iter_log = int(num_iters / 2)
    iter_valid = num_iters
    iter_save = [0, num_iters - 1] + list(range(0, num_iters, int(len(train_dataset) / args.batch_size)-1))

    start_iter = 0
    start_epoch = 0
    rate = 0

    log = Logger()
    initial_logs(log, out_dir, COMMON_STRING, IDENTIFIER, SEED, initial_checkpoint,
                 args.batch_size, train_dataset, valid_dataset, optimizer, scheduler, net, iter_accum)

    kaggle = (0, 0, 0, 0)
    valid_loss = np.zeros(6, np.float32)
    train_loss = np.zeros(3, np.float32)
    batch_loss = np.zeros_like(train_loss)
    iter = 0
    i = 0

    start_timer = timer()
    while iter < num_iters:

        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        optimizer.zero_grad()
        for t, (input, truth, infor) in enumerate(train_loader):

            batch_size = len(infor)
            iter = i + start_iter
            epoch = (iter - start_iter) * batch_size / len(train_dataset) + start_epoch

            valid_loss, kaggle = check_iter(log, out_dir, start_iter, iter, iter_valid, iter_save, iter_log, train_loss,
                                            batch_loss, start_timer, rate, epoch, net, valid_loader, criterion,
                                            NUM_TASK, valid_loss, kaggle)


            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            # net.set_mode('train',is_freeze_bn=True)

            net.train()
            input = input.cuda()
            truth = [t.cuda() for t in truth]
            logit = data_parallel(net, input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, truth)

            ((2 * loss[0] + loss[1] + loss[2]) / iter_accum).backward()
            if (iter % iter_accum) == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  --------
            loss = [l.item() for l in loss]
            l = np.array([*loss, ]) * batch_size
            n = np.array([1, 1, 1]) * batch_size
            batch_loss = l / (n + 1e-8)
            sum_train_loss += l
            sum_train += n

            if iter % iter_smooth == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train[...] = 0

            print('\r', end='', flush=True)
            print(log_message(rate, iter, epoch, kaggle, valid_loss, train_loss,
                              batch_loss, iter_save, start_timer, mode='print'),
                  end='', flush=True)

            i = i + 1

        # learning rate scheduler -------------
        lr = scheduler(iter)
        if lr < 0: break
        adjust_learning_rate(optimizer, lr)

        pass  # -- end of one data loader --
    pass  # -- end of all iterations --

    log.write('\n')


run_train()
