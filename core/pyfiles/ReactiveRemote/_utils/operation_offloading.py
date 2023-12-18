import random
import numpy as np

# RandAugment_Manual or RandAugment_Auto
# For RandAugment_Manual
N = 2   # Number of augmentations
M = 9   # Augmentation magnitude

# Total Order 
# Operation = (STAGE, OP_NAME, METADATA)
OP_STAGE_ORDER = {
  "randaugment": {
    "Start": "Decode", 
    "Decode": "RandAugment_Manual", 
    "RandAugment_Manual": "RandomCrop",
    # "Decode": "RandAugment_Auto", 
    # "RandAugment_Auto": "RandomCrop", 

    "RandomCrop": "FlipAndNormalize", 
    "FlipAndNormalize": "Finish"
  },
  "autoaugment": {
    "Start": "Decode", 
    "Decode": "AutoAugment_Manual", 
    "AutoAugment_Manual": "RandomCrop",
    "RandomCrop": "FlipAndNormalize", 
    "FlipAndNormalize": "Finish"
  },
  "deepautoaugment": {
    "Start": "Decode", 
    "Decode": "DeepAutoAugment_Manual",
    "DeepAutoAugment_Manual": "CenterCrop",
    "CenterCrop": "Finish"
  }
}

RAND_AUGMENT_MANUAL_OPTIONS = [
  "AutoContrast", #1
  "Equalize", #2
  "Invert", #3
  "Rotate", #4
  "Posterize", #5
  "Solarize", #6
  "SolarizeAdd", #7
  "Color", #8
  "Contrast", #9
  "Brightness", #10
  "Sharpness", #11
  "ShearX", #12
  "ShearY", #13
  # "CutoutAbs",
  "TranslateXabs", #14
  "TranslateYabs" #15
]

DeepAUTOAUGMENT_OPTIONS = [
    ("ShearX", -0.3, 0.3),  # 0
    ("ShearY", -0.3, 0.3),  # 1
    ("TranslateX", -0.45, 0.45),  # 2
    ("TranslateY", -0.45, 0.45),  # 3
    ("Rotate", -30, 30),  # 4
    ("AutoContrast", 0, 1),  # 5
    ("Invert", 0, 1),  # 6
    ("Equalize", 0, 1),  # 7
    ("Solarize", 0, 256),  # 8
    ("Posterize", 4, 8),  # 9
    ("Contrast", 0.1, 1.9),  # 10
    ("Color", 0.1, 1.9),  # 11
    ("Brightness", 0.1, 1.9),  # 12
    ("Sharpness", 0.1, 1.9),  # 13
    # (Cutout, 0, 0.2),  # 14
    # ("RandResizeCrop_imagenet", 0., 1.0), # 15
    # (RandCutout60, 0., 1.0) # 16
]
# From DALI's ImageNetPolicy - AutoAugment
POLICY_LIST = [
  [("Equalize", 0.8, None), ("ShearY", 0.8, 4)],
  [("Color", 0.4, 9), ("Equalize", 0.6, None)],
  [("Color", 0.4, 1), ("Rotate", 0.6, 8)],
  [("Solarize", 0.8, 3), ("Equalize", 0.4, None)],
  [("Solarize", 0.4, 2), ("Solarize", 0.6, 2)],
  [("Color", 0.2, 0), ("Equalize", 0.8, None)],
  [("Equalize", 0.4, None), ("SolarizeAdd", 0.8, 3)],
  [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],
  [("Color", 0.6, 1), ("Equalize", 1.0, None)],
  [("Invert", 0.4, None), ("Rotate", 0.6, 0)],
  [("Equalize", 1.0, None), ("ShearY", 0.6, 3)],
  [("Color", 0.4, 7), ("Equalize", 0.6, None)],
  [("Posterize", 0.4, 6), ("AutoContrast", 0.4, None)],
  [("Solarize", 0.6, 8), ("Color", 0.6, 9)],
  [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],
  [("Solarize", 1.0, 7), ("TranslateYabs", 0.8, 9)],
  [("Solarize", 0.8, 4)],
  [("ShearY", 0.8, 0), ("Color", 0.6, 4)],
  [("Color", 1.0, 0), ("Solarize", 0.6, 2)],
  [("Equalize", 0.8, None)],
  [("Equalize", 1.0, None), ("AutoContrast", 0.6, None)],
  [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],
  [("Posterize", 0.8, 2), ("Solarize", 0.6, 10)],
  [("Solarize", 0.6, 8), ("Equalize", 0.6, None)],
  [("Color", 0.8, 6), ("Solarize", 0.4, 5)]
]

AUG_NAME_LIST = [
  ('AutoContrast', 0, 1),
  ('Equalize', 0, 1),
  ('Invert', 0, 1),
  ('Rotate', 0, 30),
  ('Posterize', 0, 4),
  ('Solarize', 0, 256),
  ('SolarizeAdd', 0, 110),
  ('Color', 0.1, 1.9),
  ('Contrast', 0.1, 1.9),
  ('Brightness', 0.1, 1.9),
  ('Sharpness', 0.1, 1.9),
  ('ShearX', 0., 0.3),
  ('ShearY', 0., 0.3),
  # ('CutoutAbs', 0, 40),
  ('TranslateXabs', 0., 100),
  ('TranslateYabs', 0., 100),
]


def start_op():
  return ("Start", "Start")


def is_start(op_type):
  return op_type[0] == "Start"


def get_op_name(op_type):
  return op_type[1]


def next_op(stage_order, op_type):
  '''Returns next operation. '''
  # RandAugment Manual is still in progress
  if op_type[0] == "RandAugment_Manual" and op_type[2] is not None:
    return (op_type[0], op_type[2][0], None if len(op_type[2]) == 1 else op_type[2][1:])

  nxt = stage_order[op_type[0]]
  # RandAugment Manual, create sequence of augmentations
  if nxt == "RandAugment_Manual":
    ops = random.choices(RAND_AUGMENT_MANUAL_OPTIONS, k=N)
    return (nxt, ops[0], None if len(ops) == 1 else ops[1:])
  elif nxt == "AutoAugment_Manual":
    policy_id = random.randint(0, len(POLICY_LIST) - 1)
    policy_name = f"Policy {policy_id}"
    return (nxt, policy_name)
  # AutoAugment Manual, create sequence of augmentations
  elif nxt == "DeepAutoAugment_Manual":
    # raise NotImplementedError()
    policy_data = np.load('/home/mansur/Manticore/manticore/core/pyfiles/Augmentations/policy_port/policy_DeepAA.npz')
    policy_probs = policy_data['policy_probs']
    # l_ops = policy_data['l_ops']
    # l_mags = policy_data['l_mags']
    # ops = policy_data['ops']
    # mags = policy_data['mags']
    # op_names = policy_data['op_names']
    # for k_policy in policy_probs:
    #   k_samp = random.choices(range(len(k_policy)), weights=k_policy, k=1)[0]
    #   op, mag = np.squeeze(ops[k_samp]), np.squeeze(mags[k_samp]).astype(np.float32)/float(l_mags-1)
    #   op_name = op_names[op].split(':')[0]
      # How to add the chain? (10.04)
    return (nxt, policy_probs)
  elif nxt == "Augmix_Manual":
    raise NotImplementedError()
  return (nxt, nxt)


def is_finish(op_type):
  return op_type[0] == "Finish"


def conditional_offloading_cpu(op_type):
  return 0


def conditional_offloading_gpu(op_type):
  # if op_type[0] == "RandAugment_Manual" and op_type[2] == None:
  #   return 0
  return -1