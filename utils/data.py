from fastai.vision import *
import os
from os import environ
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import imageio
import cv2
import albumentations
from file import *


class BengaliDataset(Dataset):

    def __init__(self, df, data_path, augment=None):
        self.image_ids = df['image_id'].values
        self.grapheme_roots = df['grapheme_root'].values
        self.vowel_diacritics = df['vowel_diacritic'].values
        self.consonant_diacritics = df['consonant_diacritic'].values

        self.data_path = data_path
        self.augment = augment

    def __str__(self):
        string = ''
        string += '\tlen = %d\n' % len(self)
        string += '\n'
        return string

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # print(index)
        image_id = self.image_ids[index]
        grapheme_root = self.grapheme_roots[index]
        vowel_diacritic = self.vowel_diacritics[index]
        consonant_diacritic = self.consonant_diacritics[index]

        image_id = os.path.join(self.data_path, image_id + '.png')

        image = cv2.imread(image_id, 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.astype(np.float32) / 255
        label = [grapheme_root, vowel_diacritic, consonant_diacritic]

        infor = Struct(
            index=index,
            image_id=image_id,
        )

        if self.augment is None:
            return image, label, infor
        else:
            # return self.augment(image, label, infor)  # For Heng Augment
            image = self.augment(image=image)["image"]
            return image, label, infor  # For Albumentations


def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    input = np.stack(input)
    # input = input[...,::-1].copy()
    input = input.transpose(0, 3, 1, 2)

    label = np.stack(label)

    # ----
    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(label).long()
    truth0, truth1, truth2 = truth[:, 0], truth[:, 1], truth[:, 2]
    truth = [truth0, truth1, truth2]
    return input, truth, infor


# ======================
# DATA AUGMENTATION

def data_augmentation_selector(da_policy, img_size, crop_size):
    if da_policy is None or da_policy == "None":
        return da_policy_None(img_size)
    elif da_policy == "da7":
        return da_policy_DA7(img_size, crop_size)
    elif da_policy == "da8":
        return da_policy_DA8(img_size, crop_size)
    assert False, "Unknown Data Augmentation Policy: {}".format(da_policy)


def da_policy_None(img_size):
    train_da = albumentations.Compose([
    ])

    val_da = albumentations.Compose([
        albumentations.Resize(img_size, img_size)
    ])

    return train_da, val_da

# def da_policy_DA1():
#     return get_transforms(do_flip=False, max_warp=0.2, max_zoom=1.25)
#
# def da_policy_DA2():
#     return get_transforms(do_flip=False, max_warp=0.25, max_zoom=1.25, p_lighting=0, max_rotate=17)
#
# def da_policy_DA3():
#     additional_aug = [*
#         zoom_crop(scale=(0.75, 1.25), do_rand=False),
#     ]
#     return get_transforms(do_flip=False, max_warp=0.25, max_zoom=1.25, p_lighting=0, max_rotate=17, xtra_tfms=additional_aug)
#
# def da_policy_DA4():
#     additional_aug = [*
#         zoom_crop(scale=(0.85, 1.15), do_rand=True),
#     ]
#     return get_transforms(do_flip=False, max_warp=0.25, max_zoom=1.25, p_lighting=0, max_rotate=17, xtra_tfms=additional_aug)
#
#
# def da_policy_DA5():
#     additional_aug = [*
#         zoom_crop(scale=(0.85, 1.15), do_rand=True),
#         cutout(n_holes=(1, 2), length=(32, 84), p=.5),
#     ]
#     return get_transforms(do_flip=False, max_warp=0.25, max_zoom=1.25, p_lighting=0, max_rotate=17, xtra_tfms=additional_aug)
#
# def da_policy_DA6():
#     additional_aug = [*
#         zoom_crop(scale=(0.85, 1.15), do_rand=True),
#         cutout(n_holes=(1, 2), length=(32, 84), p=.5),
#         brightness(change=(0.33, 0.68), p=.5),
#     ]
#     return get_transforms(do_flip=False, max_warp=0.25, max_zoom=1.25, max_rotate=17, xtra_tfms=additional_aug)


def da_policy_DA7(img_size, crop_size):
    # additional_aug = [*
    #     zoom_crop(scale=(0.85, 1.15), do_rand=True),
    #     cutout(n_holes=(1, 2), length=(32, 84), p=.5),
    #     brightness(change=(0.33, 0.68), p=.5),
    #     contrast(scale=(0.7, 1.4), p=.5),
    # ]
    # return get_transforms(do_flip=False, max_warp=0.25, max_zoom=1.25, max_rotate=17, xtra_tfms=additional_aug)

    train_da = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.RandomCrop(p=1, height=crop_size, width=crop_size),
        albumentations.CoarseDropout(p=0.5, min_holes=1, max_holes=2,
                                     min_width=32, min_height=32, max_width=84, max_height=84),
        albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.25),
        albumentations.Rotate(p=0.5, limit=17)
    ])

    val_da = albumentations.Compose([
        albumentations.Resize(crop_size, crop_size)
    ])

    return train_da, val_da


def da_policy_DA8(img_size, crop_size):
    train_da = albumentations.Compose([
        albumentations.Resize(img_size, img_size),
        albumentations.RandomCrop(p=1, height=crop_size, width=crop_size),
        albumentations.CoarseDropout(p=0.5, min_holes=1, max_holes=2,
                                     min_width=32, min_height=32, max_width=84, max_height=84),
        albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.25),
        albumentations.Rotate(p=0.5, limit=17),
        albumentations.GridDistortion(p=0.5, distort_limit=0.2, num_steps=10)
    ])

    val_da = albumentations.Compose([
        albumentations.Resize(crop_size, crop_size)
    ])

    return train_da, val_da


# def da_policy_DA9():
#     additional_aug = [*
#         zoom_crop(scale=(0.85, 1.15), do_rand=True),
#         cutout(n_holes=(1, 2), length=(50, 112), p=.55),
#         brightness(change=(0.33, 0.68), p=.5),
#         contrast(scale=(0.7, 1.4), p=.5),
#     ]
#     return get_transforms(do_flip=False, max_warp=0.25, max_zoom=1.25, max_rotate=17, xtra_tfms=additional_aug)


# === HENG DATA AUGMENTATION


##############################################################

def tensor_to_image(tensor):
    image = tensor.data.cpu().numpy()
    image = image.transpose(0, 2, 3, 1)
    # image = image[...,::-1]
    return image


##############################################################

def do_random_crop_rotate_rescale(
        image,
        mode={'rotate': 10, 'scale': 0.1, 'shift': 0.1}
):
    dangle = 0
    dscale_x, dscale_y = 0, 0
    dshift_x, dshift_y = 0, 0

    for k, v in mode.items():
        if 'rotate' == k:
            dangle = np.random.uniform(-v, v)
        elif 'scale' == k:
            dscale_x, dscale_y = np.random.uniform(-1, 1, 2) * v
        elif 'shift' == k:
            dshift_x, dshift_y = np.random.uniform(-1, 1, 2) * v
        else:
            raise NotImplementedError

    # ----

    height, width = image.shape[:2]

    cos = np.cos(dangle / 180 * PI)
    sin = np.sin(dangle / 180 * PI)
    sx, sy = 1 + dscale_x, 1 + dscale_y  # 1,1 #
    tx, ty = dshift_x * width, dshift_y * height

    src = np.array(
        [[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]],
        np.float32)
    src = src * [sx, sy]
    x = (src * [cos, -sin]).sum(1) + width / 2 + tx
    y = (src * [sin, cos]).sum(1) + height / 2 + ty
    src = np.column_stack([x, y])

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s, d)
    image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))

    return image


def do_random_log_contast(image, gain=[0.70, 1.30]):
    gain = np.random.uniform(gain[0], gain[1], 1)
    inverse = np.random.choice(2, 1)

    if inverse == 0:
        image = gain * np.log(image + 1)
    else:
        image = gain * (2 ** image - 1)

    image = np.clip(image, 0, 1)
    return image


# https://github.com/albumentations-team/albumentations/blob/8b58a3dbd2f35558b3790a1dbff6b42b98e89ea5/albumentations/augmentations/transforms.py
def do_grid_distortion(image, distort=0.25, num_step=10):
    # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    distort_x = [1 + random.uniform(-distort, distort) for i in range(num_step + 1)]
    distort_y = [1 + random.uniform(-distort, distort) for i in range(num_step + 1)]

    # ---
    height, width = image.shape[:2]
    xx = np.zeros(width, np.float32)
    step_x = width // num_step

    prev = 0
    for i, x in enumerate(range(0, width, step_x)):
        start = x
        end = x + step_x
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + step_x * distort_x[i]
        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    yy = np.zeros(height, np.float32)
    step_y = height // num_step

    prev = 0
    for idx, y in enumerate(range(0, height, step_y)):
        start = y
        end = y + step_y
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + step_y * distort_y[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(1, 1, 1))

    return image


# ##---
# #https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
def do_random_contast(image, alpha=[0, 1]):
    beta = 0
    alpha = random.uniform(*alpha) + 1
    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image, 0, 1)
    return image


# In[30]:


################################################################################################

def train_augment(image, label, infor):
    if np.random.rand() < 0.5:
        image = do_random_crop_rotate_rescale(image, mode={'rotate': 17.5, 'scale': 0.25, 'shift': 0.08})
    if np.random.rand() < 0.5:
        image = do_grid_distortion(image, distort=0.20, num_step=10)
    return image, label, infor


def valid_augment(image, label, infor):
    return image, label, infor