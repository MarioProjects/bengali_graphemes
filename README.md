# Bengali.AI Handwritten Grapheme Classification 
## 33 Place Solution (Top 2%)

### Description
Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant business and educational interest in developing AI that can optically recognize images of the language handwritten. This challenge hopes to improve on approaches to Bengali recognition.

Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).

Bangladesh-based non-profit Bengali.AI is focused on helping to solve this problem. They build and release crowdsourced, metadata-rich datasets and open source them through research competitions. Through this work, Bengali.AI hopes to democratize and accelerate research in Bengali language technologies and to promote machine learning education.

For this competition, you’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.

### Solution

The model is based on SE-ResNext101.

#### Data Augmentation
Applying augmentations by the following order, implemented by [albumentations](https://albumentations.readthedocs.io/en/latest/api/augmentations.html)

  - ShiftScaleRotate(scale_limit=0.15, rotate_limit=0)
  - Resize(crop_size, crop_size)
  - CoarseDropout(min_holes=1, max_holes=2, min_width=16, min_height=16, max_width=64, max_height=64)
  - RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25)
  - Rotate(limit=17)

#### Optimization
Batch size: 50

First Phase: Initial Train
  - Optimizer : Over9000
  - epoch: 29
  - learning schedule: OneCycleLR
    - learning rate: max=0.01, min=0.002
    
Second Phase: Mixup & Cutmix  
  - Optimizer : SGD
  - epoch: 100
  - learning schedule: Constant
    - learning rate: 0.01
  - Mixup: alpha=0.4, prob=0.55
  - Cutmix: alpha=1.0, prob=0.35
  
Third Phase: SWA
  - Optimizer : SGD
  - epoch: 100
  - learning schedule: Constant
    - learning rate: 0.0457
  - SWA: freq=1

[Leadeboard](https://www.kaggle.com/c/bengaliai-cv19/leaderboard)