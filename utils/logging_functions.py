import torch
from utils.training import *

def initial_logs(log, out_dir, COMMON_STRING, IDENTIFIER, SEED, initial_checkpoint,
                 batch_size, train_dataset, valid_dataset, optimizer, scheduler, net, iter_accum):

    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    assert (len(train_dataset) >= batch_size)
    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    log.write('net=%s\n' % (type(net)))
    log.write('\n')

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('scheduler\n  %s\n' % (scheduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n' % (batch_size, iter_accum))
    log.write(
        '                    |----------------------- VALID------------------------------------|------- TRAIN/BATCH -----------\n')
    log.write(
        'rate    iter  epoch | kaggle                    | loss               acc              | loss             | time       \n')
    log.write(
        '----------------------------------------------------------------------------------------------------------------------\n')


def initial_logs_simple(log, out_dir, COMMON_STRING, IDENTIFIER, SEED, initial_checkpoint,
                 batch_size, train_dataset, valid_dataset, optimizer, scheduler, net, epochs, grad_clip):

    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('SEED     = %u\n' % SEED)
    log.write('out_dir  = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    assert (len(train_dataset) >= batch_size)
    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    log.write('net=%s\n' % (type(net)))
    log.write('\n')

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('scheduler\n  %s\n' % (scheduler))
    log.write('\nepochs=%d\n' % (epochs))
    log.write('batch_size=%d\n' % (batch_size))
    if grad_clip != 9999:
        log.write('grad_clip=%d\n' % (grad_clip))
    else: log.write('grad_clip=None\n')
    log.write('\n')

    ## start training here! ##############################################
    log.write('** START TRAINING HERE! **\n\n\n')
    log.write(
        '                |----------------------- VALID------------------------------------|------- TRAIN/BATCH ----\n')
    log.write(
        'epoch     rate  | kaggle                    | loss               acc              | loss   | time       \n')
    log.write(
        '-----------------------------------------------------------------------------------------------------------\n')

def check_iter(log, out_dir, start_iter, iter, iter_valid, iter_save, iter_log, train_loss, batch_loss,
               start_timer, rate, epoch, net, valid_loader, criterion, NUM_TASK, valid_loss, kaggle):

    if iter % iter_valid == 0 and epoch != 0:
        print("Valid")
        valid_loss, kaggle = do_valid(net, valid_loader, criterion, NUM_TASK)  #
        pass

    if iter % iter_log == 0:
        print('\r', end='', flush=True)
        log.write(log_message(rate, iter, epoch, kaggle, valid_loss, train_loss,
                              batch_loss, iter_save, start_timer, mode='log'))
        log.write('\n')

    if iter in iter_save:
        torch.save({
            # 'optimizer': optimizer.state_dict(),
            'iter': iter,
            'epoch': epoch,
        }, out_dir + '/checkpoint/%08d_optimizer.pth' % (iter))
        if iter != start_iter:
            torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (iter))
            pass

    return valid_loss, kaggle