from utils.rate import *
import random
import os
import torch

# ---------------------------------------------------------------------------------
COMMON_STRING = '@%s:\n' % os.path.basename(__file__)

if 1:
    SEED = 2019  # int(time.time())
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    COMMON_STRING += 'set random seed\n'
    COMMON_STRING += '\tSEED = %d\n' % SEED

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    COMMON_STRING += 'set cuda environment\n'
    COMMON_STRING += '\ttorch.__version__              = %s\n' % torch.__version__
    COMMON_STRING += '\ttorch.version.cuda             = %s\n' % torch.version.cuda
    COMMON_STRING += '\ttorch.backends.cudnn.version() = %s\n' % torch.backends.cudnn.version()
    try:
        COMMON_STRING += '\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n' % os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        COMMON_STRING += '\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
        NUM_CUDA_DEVICES = 1

    COMMON_STRING += '\ttorch.cuda.device_count()      = %d\n' % torch.cuda.device_count()
    # print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())

COMMON_STRING += '\n'

# ---------------------------------------------------------------------------------
## useful : http://forums.fast.ai/t/model-visualization/12365/2


if __name__ == '__main__':
    print(COMMON_STRING)
