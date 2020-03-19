#!/usr/bin/env python
# coding: utf-8

from models import *
from utils.arguments.train_arguments import *
from utils.data import *
from utils.logging_functions import *
from utils.kaggle import *
from utils.onecyclelr import OneCycleLR
from utils.radam import *
from torchcontrib.optim import SWA

train_df, valid_df = train_test_split(df, test_size=args.validation_size, shuffle=True, random_state=2019)

TASK_NAME = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
NUM_TASK = len(TASK_NAME)

train_augment, valid_augment = data_augmentation_selector(args.data_augmentation, args.img_size, args.crop_size)

out_dir = args.output_dir
os.makedirs(out_dir + '/checkpoint', exist_ok=True)

train_dataset = BengaliDataset(df=train_df, data_path=data_root, augment=train_augment)
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size,
                          drop_last=True, num_workers=4, pin_memory=True, collate_fn=null_collate)

valid_dataset = BengaliDataset(df=valid_df, data_path=data_root, augment=valid_augment)
valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=args.batch_size,
                          drop_last=False, num_workers=4, pin_memory=True, collate_fn=null_collate)

net = model_selector(args.model_name, args.head_name, [168, 11, 7], pretrained=args.pretrained).cuda()
load_from_checkpoint(net, args.model_checkpoint)
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

optimizer = select_optimizer(args.optimizer, net, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
if args.apply_swa: optimizer = SWA(optimizer, swa_start=0, swa_freq=1, swa_lr=args.min_lr)
scheduler = select_scheduler(args.scheduler, optimizer, args.min_lr, args.max_lr, epochs=args.epochs, decay=args.scheduler_decay, step=args.scheduler_step)
criterion = select_criterion(args.criterion)

log = Logger()
initial_logs_simple(log, out_dir, COMMON_STRING, IDENTIFIER, SEED, args.model_checkpoint,
             args.batch_size, train_dataset, valid_dataset, optimizer, scheduler, net, args.epochs,
             args.mixup_prob, args.mixup_alpha, args.cutmix_prob, args.cutmix_alpha, args.grad_clipping)

start_timer, best_metric = timer(), 0
for epoch in range(args.epochs):
    train_loss = train(net, train_loader, optimizer, criterion, args.mixup_prob, args.mixup_alpha,
                       args.cutmix_prob, args.cutmix_alpha, args.grad_clipping)
    valid_loss, kaggle = valid(net, valid_loader, criterion, NUM_TASK)

    show_simple_stats(log, epoch, optimizer, start_timer, kaggle, train_loss, valid_loss)

    if kaggle[1] > best_metric:
        best_metric = kaggle[1]
        torch.save(net.state_dict(), out_dir + '/checkpoint/best_model.pth')

    # learning rate scheduler -------------
    if epoch < args.epochs-1:  # Prevent step on last epoch
        scheduler_step(args.scheduler, scheduler, optimizer, epoch)

torch.save(net.state_dict(), out_dir + '/checkpoint/last_model.pth')

if args.apply_swa:
    torch.save(optimizer.state_dict(), args.output_dir + "/optimizer_" + args.model_name + "_last_before_swap.pt")
    optimizer.swap_swa_sgd()
    optimizer.bn_update(train_loader, net, device='cuda')
    torch.save(net.state_dict(), args.output_dir + "/model_" + args.model_name + "_last_bn_update.pt")

log.write('\n')
