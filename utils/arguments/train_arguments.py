import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(
    description='Kaggle Bengali Classification',
    formatter_class=SmartFormatter)

parser.add_argument('--slack_resume', action='store_true', help='Send Slack message when train finish')
# parser.add_argument('--clip_grad', action='store_true', help='Wheter clip grad or learner to 1.0')
parser.add_argument('--pretrained', action='store_true', help='Wheter use pretrained on Imagenet model or not')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')
parser.add_argument('--epochs', type=int, default=40, help='Total number epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--min_lr', type=float, default=0.002, help='Min Learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='Max Learning rate')
parser.add_argument('--criterion', type=str, default='combined_crossentropy', help='Criterion for training')
parser.add_argument('--model_name', type=str, default='densenet121', help='Model name for training')
parser.add_argument('--head_name', type=str, default='initial_head', help='Head name for training')
parser.add_argument('--optimizer', type=str, default='over9000', help='Optimizer for training')
parser.add_argument('--scheduler', type=str, default='steps', help='LR Scheduler for training')
parser.add_argument('--loss', type=str, default='standard_loss', help='Loss for training')

parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')
parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Alpha for Mixup. If 0, mixup is not applied')
parser.add_argument('--crop_size', type=int, default=224, help='Center crop squared size')
parser.add_argument('--img_size', type=int, default=224, help='Final img squared size')

parser.add_argument('--model_checkpoint', type=str, default="", help='Where is the model checkpoint saved')
# parser.add_argument('--defrost_epoch', type=int, default=-1, help='Number of epochs to defrost the model')
parser.add_argument('--validation_size', type=float, default=0.2, help='Validation partition size')

parser.add_argument('--output_dir', type=str, default='results/new_logs', help='Where progress will be saved')
parser.add_argument('--additional_info', type=str, default='', help='Additional info appended to the saving path')


try:
    args = parser.parse_args()
except:
    print("Working with Jupyter notebook! (Default Arguments)")
    args = parser.parse_args("")

if args.output_dir == "results/new_logs":
    args.output_dir = "results/{}/{}_{}_{}_{}_lr{}_{}to{}_mixup{}_DA{}{}_MinLr{}_MaxLr{}_{}".format(
                                                                                args.model_name,
                                                                                args.head_name,
                                                                                args.model_name, args.criterion,
                                                                                args.optimizer, args.learning_rate,
                                                                                args.img_size, args.crop_size,
                                                                                args.mixup_alpha,
                                                                                args.data_augmentation,
                                                                                args.additional_info,
                                                                                args.min_lr, args.max_lr,
                                                                                args.scheduler)

    if args.pretrained: args.output_dir = args.output_dir + "_PRETRAINED"
