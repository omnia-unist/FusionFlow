import Augmix.augmentations
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder 
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, aug_severity=1,aug_prob_coeff=1.,mixture_depth=-1, mixture_width=3,no_jsd=False, all_ops=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    self.all_ops = all_ops
    self.aug_prob_coeff = aug_prob_coeff
    self.mixture_width = mixture_width
    self.mixture_depth = mixture_depth
    self.aug_severity = aug_severity
    
  def __getitem__(self, i):
    x, y = self.dataset[i]

    if self.no_jsd:
      out=self.aug(x)
      return out, y
    else:
      im_tuple = (self.preprocess(x), self.aug(x),
                  self.aug(x))
      return im_tuple, y

  def aug(self, image):
    """Perform AugMix augmentations and compute mixture.

    Args:
        image: PIL.Image input image
    Returns:
        mixed: Augmented and mixed image.
    """
    aug_list = Augmix.augmentations.augmentations
    if self.all_ops:
        aug_list = Augmix.augmentations.augmentations_all

    ws = np.float32(
        np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
    m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

    mix = torch.zeros_like(self.preprocess(image))
    for i in range(self.mixture_width):
        image_aug = image.copy()
        depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, self.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * self.preprocess(image_aug)

    mixed = (1 - m) * self.preprocess(image) + m * mix
    return mixed

  def __len__(self):
    return len(self.dataset)