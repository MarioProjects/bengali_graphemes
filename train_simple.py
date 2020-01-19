#!/usr/bin/env python
# coding: utf-8

# This kernel is based on [Heng's Starter code](https://www.kaggle.com/c/bengaliai-cv19/discussion/123757).
# I have already published a [kernel](https://www.kaggle.com/bibek777/heng-starter-inference-kernel)
# doing the inference using his models. In this kernel, we will use his codes to do the training.

# Any results you write to the current directory are saved as output.


# Add heng's code to our envionment and import the modules


from models import *
from utils.arguments.train_arguments import *
from utils.data import *
from utils.logging_functions import *
from utils.kaggle import *
from utils.onecyclelr import OneCycleLR
from utils.radam import *

# Heng uses his own version of datasplit(which I have uploaded [here](https://www.kaggle.com/bibek777/hengdata)).
# I tried using it but get memory error, maybe it's too large to load. So I have edited the dataloader in his code
# and use different split for train and valid dataset. The codes/ideas for dataloader is taken from this
# [kernel](https://www.kaggle.com/backaggle/catalyst-baseline). Also the dataset used in this kernel is taken
# from [here](https://www.kaggle.com/pestipeti/bengaliai), uploaded by Peter


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
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

if args.scheduler == "one_cycle_lr":
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=args.min_lr, momentum=0.0, weight_decay=0.0)
    optimizer = Over9000(filter(lambda p: p.requires_grad, net.parameters()))
    scheduler = OneCycleLR(optimizer, num_steps=args.epochs, lr_range=(args.min_lr, args.max_lr))
else:
    scheduler = DecayScheduler(base_lr=args.learning_rate, decay=0.1, step=30)

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

log = Logger()
initial_logs_simple(log, out_dir, COMMON_STRING, IDENTIFIER, SEED, args.model_checkpoint,
             args.batch_size, train_dataset, valid_dataset, optimizer, scheduler, net, args.epochs)

start_timer, best_metric = timer(), 0
for epoch in range(args.epochs):
    train_loss = train(net, train_loader, optimizer, criterion)
    valid_loss, kaggle = valid(net, valid_loader, criterion, NUM_TASK)

    show_simple_stats(log, epoch, optimizer, start_timer, kaggle, train_loss, valid_loss)

    if kaggle[1] > best_metric:
        best_metric = kaggle[1]
        torch.save(net.state_dict(), out_dir + '/checkpoint/best_model.pth')

    # learning rate scheduler -------------
    scheduler_step(args.scheduler, scheduler, optimizer, epoch)

torch.save(net.state_dict(), out_dir + '/checkpoint/last_model.pth')
log.write('\n')
