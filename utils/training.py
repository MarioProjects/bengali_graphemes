from slackclient import SlackClient
from os import environ
from utils.csvlogger import *
from file import *
from utils.kaggle import *
from utils.radam import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def slack_message(message, user_slack_token="SYS", channel="experimentos"):
    if user_slack_token == "SYS":
        if environ.get('SLACK_TOKEN') is not None:
            user_slack_token = environ.get('SLACK_TOKEN')
        else:
            assert False, "Please set the environment variable SLACK_TOKEN.!"

    token = user_slack_token
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel,
                text=message, username='Experiments Bot',
                icon_emoji=':robot_face:')


# ====== HENG Utils

def log_message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, iter_save, start_timer, mode='print'):
    print(iter)
    if iter == 1 or iter == 0: return ""

    if mode == ('print'):
        asterisk = ' '
        loss = batch_loss
    if mode == ('log'):
        asterisk = '*' if iter in iter_save else ' '
        loss = train_loss

    text = '%0.5f %5.1f%s %4.1f | ' % (rate, iter / 1000, asterisk, epoch,) + '%0.3f : %0.3f %0.3f %0.3f | ' % (
        kaggle[1], *kaggle[0]) + '%4.2f, %4.2f, %4.2f : %4.2f, %4.2f, %4.2f | ' % (
               *valid_loss,) + '%4.2f, %4.2f, %4.2f |' % (*loss,) + '%s' % (time_to_str((timer() - start_timer), 'min'))

    return text


# ------------------------------------
def do_valid(net, valid_loader, criterion, NUM_TASK):
    valid_loss = np.zeros(6, np.float32)
    valid_num = np.zeros_like(valid_loss)

    valid_probability = [[], [], [], ]
    valid_truth = [[], [], [], ]

    for t, (input, truth, infor) in enumerate(valid_loader):

        # if b==5: break
        batch_size = len(infor)

        net.eval()
        input = input.cuda()
        truth = [t.cuda() for t in truth]

        with torch.no_grad():
            # logit = data_parallel(net, input)  # net(input)
            logit = net(input)  # net(input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, truth)
            correct = metric(probability, truth)

        # ---
        loss = [l.item() for l in loss]
        l = np.array([*loss, *correct, ]) * batch_size
        n = np.array([1, 1, 1, 1, 1, 1]) * batch_size
        valid_loss += l
        valid_num += n

        # ---
        for i in range(NUM_TASK):
            valid_probability[i].append(probability[i].data.cpu().numpy())
            valid_truth[i].append(truth[i].data.cpu().numpy())

        # print(valid_loss)
        print('\r %8d /%d' % (valid_num[0], len(valid_loader.dataset)), end='', flush=True)

        pass  # -- end of one data loader --
    assert (valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss / (valid_num + 1e-8)

    # ------
    for i in range(NUM_TASK):
        valid_probability[i] = np.concatenate(valid_probability[i])
        valid_truth[i] = np.concatenate(valid_truth[i])
    recall, avgerage_recall = compute_kaggle_metric(valid_probability, valid_truth)

    return valid_loss, (recall, avgerage_recall)


def logit_to_probability(logit):
    probability = []
    for l in logit:
        p = F.softmax(l, 1)
        probability.append(p)
    return probability


#########################################################################

def select_criterion(criterion_name):
    if criterion_name == "combined_crossentropy":
        return cross_entropy_criterion
    else:
        assert False, "Unknown criterion: {}".format(criterion_name)


def cross_entropy_criterion(logit, truth):
    loss = []
    for l, t in zip(logit, truth):
        e = F.cross_entropy(l, t)
        loss.append(e)

    return loss


def select_optimizer(optimizer_name, net, lr, momentum=0.0, weight_decay=0.0):
    if optimizer_name == "over9000":
        return Over9000(filter(lambda p: p.requires_grad, net.parameters()))
    elif optimizer_name == "sgd":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                               lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        assert False, "Unknown optimizer: {}".format(optimizer_name)


def select_scheduler(scheduler_name, optimizer, min_lr, max_lr, epochs=100, decay=0.1, step=25):
    from utils.onecyclelr import OneCycleLR
    if scheduler_name == "one_cycle_lr":
        return OneCycleLR(optimizer, num_steps=epochs, lr_range=(min_lr, max_lr))
    elif scheduler_name == "steps":
        return DecayScheduler(base_lr=min_lr, decay=decay, step=step)
    else:
        assert False, "Unknown scheduler: {}".format(scheduler_name)


# https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
def metric(probability, truth):
    correct = []
    for p, t in zip(probability, truth):
        p = p.data.cpu().numpy()
        t = t.data.cpu().numpy()
        y = p.argmax(-1)
        c = np.mean(y == t)
        correct.append(c)

    return correct


# ## Loss

# Cross entropy loss is applied independently to each part of the prediction and the result is summed with the corresponding weight.

def loss_selector(loss_name):
    print(loss_name)
    if loss_name is None:
        assert False, "Unknown Loss: {}".format(loss_name)
    elif loss_name == "standard_loss":
        return Loss_combine()
    elif loss_name == "reduction_loss":
        return Loss_combine_reduction()
    assert False, "Unknown Loss: {}".format(loss_name)


class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        x1, x2, x3 = input
        y = target.long()
        return 2.0 * F.cross_entropy(x1, y[:, 0]) + F.cross_entropy(x2, y[:, 1]) + F.cross_entropy(x3, y[:, 2])


class Loss_combine_reduction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, reduction='mean'):
        # if reduction not in ["mean", "sum"]: assert False, "Unknown reduction: {}".format(reduction)
        x1, x2, x3 = input
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = target.long()
        return 0.5 * F.cross_entropy(x1, y[:, 0], reduction=reduction) + 0.25 * F.cross_entropy(x2, y[:, 1],
                                                                                                reduction=reduction) + \
               0.25 * F.cross_entropy(x3, y[:, 2], reduction=reduction)


# The code below computes the competition metric and recall macro metrics for individual components of the prediction. The code is partially borrowed from fast.ai.

class Metric_idx(Callback):
    def __init__(self, idx, average='macro'):
        super().__init__()
        self.idx = idx
        self.n_classes = 0
        self.average = average
        self.cm = None
        self.eps = 1e-9

    def on_epoch_begin(self, **kwargs):
        self.tp = 0
        self.fp = 0
        self.cm = None

    def on_batch_end(self, last_output: Tensor, last_target: Tensor, **kwargs):
        last_output = last_output[self.idx]
        last_target = last_target[:, self.idx]
        preds = last_output.argmax(-1).view(-1).cpu()
        targs = last_target.long().cpu()

        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
            self.x = torch.arange(0, self.n_classes)
        cm = ((preds == self.x[:, None]) & (targs == self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None:
            self.cm = cm
        else:
            self.cm += cm

    def _weights(self, avg: str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn(
                "average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1:
                return Tensor([0, 1])
            else:
                return Tensor([1, 0])
        elif avg == "micro":
            return self.cm.sum(dim=0) / self.cm.sum()
        elif avg == "macro":
            return torch.ones((self.n_classes,)) / self.n_classes
        elif avg == "weighted":
            return self.cm.sum(dim=1) / self.cm.sum()

    def _recall(self):
        rec = torch.diag(self.cm) / (self.cm.sum(dim=1) + self.eps)
        if self.average is None:
            return rec
        else:
            if self.average == "micro":
                weights = self._weights(avg="weighted")
            else:
                weights = self._weights(avg=self.average)
            return (rec * weights).sum()

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._recall())


Metric_grapheme = partial(Metric_idx, 0)
Metric_vowel = partial(Metric_idx, 1)
Metric_consonant = partial(Metric_idx, 2)


class Metric_tot(Callback):
    def __init__(self):
        super().__init__()
        self.grapheme = Metric_idx(0)
        self.vowel = Metric_idx(1)
        self.consonant = Metric_idx(2)

    def on_epoch_begin(self, **kwargs):
        self.grapheme.on_epoch_begin(**kwargs)
        self.vowel.on_epoch_begin(**kwargs)
        self.consonant.on_epoch_begin(**kwargs)

    def on_batch_end(self, last_output: Tensor, last_target: Tensor, **kwargs):
        self.grapheme.on_batch_end(last_output, last_target, **kwargs)
        self.vowel.on_batch_end(last_output, last_target, **kwargs)
        self.consonant.on_batch_end(last_output, last_target, **kwargs)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, 0.5 * self.grapheme._recall() +
                           0.25 * self.vowel._recall() + 0.25 * self.consonant._recall())


def getMetricTot(learn):
    return 0.50 * learn.metrics[-1].grapheme._recall().item() + \
           0.25 * learn.metrics[-1].vowel._recall().item() + \
           0.25 * learn.metrics[-1].consonant._recall().item()


# fix the issue in fast.ai of saving gradients along with weights
# so only weights are written, and files are ~4 times smaller

class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."

    def __init__(self, learn: Learner, monitor: str = 'valid_loss', mode: str = 'auto',
                 every: str = 'improvement', name: str = 'bestmodel'):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every, self.name = every, name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def jump_to_epoch(self, epoch: int) -> None:
        try:
            self.learn.load(f'{self.name}_{epoch - 1}', purge=False)
            print(f"Loaded {self.name}_{epoch - 1}")
        except:
            print(f'Model {self.name}_{epoch - 1} not found.')

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every == "epoch":
            # self.learn.save(f'{self.name}_{epoch}')
            torch.save(self.learn.model.state_dict(), f'{self.name}_{epoch}.pth')
        else:  # every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                # print(f'Better model found at epoch {epoch} \
                #  with {self.monitor} value: {current}.')
                self.best = current
                # self.learn.save(f'{self.name}')
                torch.save(self.learn.model.state_dict(), f'{self.name}.pth')

    def on_train_end(self, **kwargs):
        "Load the best model."
        if self.every == "improvement" and os.path.isfile(f'{self.name}.pth'):
            # self.learn.load(f'{self.name}', purge=False)
            self.model.load_state_dict(torch.load(f'{self.name}.pth'))


# Credits: https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964
class MixUpLoss(Module):
    "Adapt the loss function `crit` to go with mixup."

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else:
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.shape) == 2 and target.shape[1] == 7:
            loss1, loss2 = self.crit(output, target[:, 0:3].long()), self.crit(output, target[:, 3:6].long())
            d = loss1 * target[:, -1] + loss2 * (1 - target[:, -1])
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit


class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."

    def __init__(self, learn: Learner, alpha: float = 0.4, stack_x: bool = False, stack_y: bool = True):
        super().__init__(learn)
        self.alpha, self.stack_x, self.stack_y = alpha, stack_x, stack_y

    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else:
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1 - lambd).view(out_shape))
        if self.stack_y:
            new_target = torch.cat([last_target.float(), y1.float(), lambd[:, None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1 - lambd)
        return {'last_input': new_input, 'last_target': new_target}

    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


# ============= LAST UTILS TRAIN SIMPLE

def train(net, train_loader, optimizer, criterion):
    net.train()
    train_loss = np.zeros(3, np.float32)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = np.zeros_like(train_loss)

    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx, (inputs, targets, infor) in enumerate(train_loader):
        batch_size = len(infor)
        inputs = inputs.cuda()
        targets = [t.cuda() for t in targets]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        (2 * loss[0] + loss[1] + loss[2]).backward()
        optimizer.step()

        loss = [l.item() for l in loss]
        l = np.array([*loss, ]) * batch_size
        n = np.array([1, 1, 1]) * batch_size
        sum_train_loss += l
        sum_train += n

    train_loss = sum_train_loss / (sum_train + 1e-12)
    return train_loss


def valid(net, valid_loader, criterion, NUM_TASK):
    valid_loss = np.zeros(6, np.float32)
    valid_num = np.zeros_like(valid_loss)

    valid_probability = [[], [], [], ]
    valid_truth = [[], [], [], ]
    net.eval()

    for batch_idx, (input, truth, infor) in enumerate(valid_loader):

        batch_size = len(infor)
        input = input.cuda()
        truth = [t.cuda() for t in truth]

        with torch.no_grad():
            # logit = data_parallel(net, input)  # net(input)
            logit = net(input)  # net(input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, truth)
            correct = metric(probability, truth)

        # ---
        loss = [l.item() for l in loss]
        l = np.array([*loss, *correct, ]) * batch_size
        n = np.array([1, 1, 1, 1, 1, 1]) * batch_size
        valid_loss += l
        valid_num += n

        # ---
        for i in range(NUM_TASK):
            valid_probability[i].append(probability[i].data.cpu().numpy())
            valid_truth[i].append(truth[i].data.cpu().numpy())

        # print(valid_loss)
        # print('\r %8d /%d' % (valid_num[0], len(valid_loader.dataset)), end='', flush=True)

    assert (valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss / (valid_num + 1e-8)

    # ------
    for i in range(NUM_TASK):
        valid_probability[i] = np.concatenate(valid_probability[i])
        valid_truth[i] = np.concatenate(valid_truth[i])
    recall, avgerage_recall = compute_kaggle_metric(valid_probability, valid_truth)

    return valid_loss, (recall, avgerage_recall)


def show_simple_stats(log, epoch, optimizer, start_timer, kaggle, train_loss, valid_loss):
    rate = get_learning_rate(optimizer)
    text = '%3.0f  %0.7f  | ' % (epoch, rate,) + '%0.3f : %0.3f %0.3f %0.3f | ' % (
        kaggle[1], *kaggle[0]) + '%4.2f, %4.2f, %4.2f : %4.2f, %4.2f, %4.2f | ' % (
               *valid_loss,) + '%4.2f, %4.2f, %4.2f |' % (*train_loss,) + '%s' % (
               time_to_str((timer() - start_timer), 'min'))

    log.write(text)
    log.write('\n')


def scheduler_step(scheduler_type, scheduler, optimizer, epoch):
    if scheduler_type == "one_cycle_lr":
        scheduler.step()
    else:
        lr = scheduler(epoch)
        if lr < 0: assert False, "Learning rate < 0! Stop train!"
        adjust_learning_rate(optimizer, lr)


def load_from_checkpoint(net, model_checkpoint):
    if model_checkpoint != "":
        print("Loading model from checkpoint...")
        # Because we used multiple GPUs training
        state_dict = torch.load(model_checkpoint, map_location=torch.device('cpu'))
        from collections import OrderedDict

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict, strict=False)
