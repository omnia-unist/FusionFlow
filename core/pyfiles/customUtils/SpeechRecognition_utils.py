import torch
import torch.utils.data as data
from torch import nn
import torchaudio
import torchaudio.functional as F
import os
import os.path
import numpy as np
import random


AUDIO_EXTENSIONS = ['.mp3', '.wav']


class AudioAugmentation(object):
  def __init__(self, noise_dataset, sample_rate, scale=0.5):
    self.scale = 0.5
    self.noises = []
    for i in range(noise_dataset.num_audios):
      noise, _ = noise_dataset[i]
      slices = noise.shape[1] // sample_rate
      for j in range(noise.shape[0]):
        self.noises.append(noise[j][:slices * sample_rate])

  def __call__(self, audio):
    # Adding Noise
    noise = [] 
    prop = []
    random_idxs = np.random.uniform(low=0, high=len(self.noises), size=audio.shape[0])

    for i in range(audio.shape[0]):
      prop.append(torch.max(audio[i]) / torch.max(self.noises[i]))
      if self.noises[i].shape[0] < audio.shape[1]:
        tmp = self.noises[i].copy()
        tmp.expand(audio.shape[1])
        noise.append(tmp)
      else:
        noise.append(self.noises[i][:audio.shape[1]])
    assert len(prop) == 1
    noise = torch.stack(noise)
    prop = torch.stack(prop)
    audio = audio + noise * prop * self.scale
    # TODO: Transformation using FFT
    audio = torch.squeeze(audio)
    audio = torch.fft.fft(audio)
    audio = audio[None, : (audio.shape[0] // 2)]
    return audio


def is_audio_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    audio = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    audio.append(item)

    return audio


class AudioFolder(data.Dataset):
    def __init__(self, root, transform=None):
        classes, class_to_idx = find_classes(root)
        audios = make_dataset(root, class_to_idx)
        if len(audios) == 0:
            raise(RuntimeError("Found 0 audios in subfolders of: " + root + "\n"
                               "Supported audio extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.audios = audios
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.num_audios = len(audios)

    def __getitem__(self, index):
        path, target = self.audios[index]
        audio, _ = torchaudio.load(path)
        if self.transform is not None:
            audio = self.transform(audio)

        return audio, target

    def get_random_audio(self):
        index = random.randint(0, self.num_audios)

        path, target = self.audios[index]
        audio, _ = torchaudio.load(path)
        if self.transform is not None:
            audio = self.transform(audio)
        return audio, target

    def __len__(self):
        return len(self.audios)


class ResidualBlock(torch.nn.Module):
  def __init__(self, input_channel, output_channel, filters, conv_num=3, activation="relu"):
    super().__init__()
    residual_block = []
    residual_block.append(torch.nn.Conv1d(input_channel, output_channel, kernel_size=1))
    residual_block.append(torch.nn.Conv1d(input_channel, output_channel, kernel_size=3, padding=1))
    residual_block.append(torch.nn.ReLU())
    for i in range(conv_num - 2):
      residual_block.append(torch.nn.Conv1d(output_channel, output_channel, kernel_size=3, padding=1))
      residual_block.append(torch.nn.ReLU())
    residual_block.append(torch.nn.Conv1d(output_channel, output_channel, kernel_size=3, padding=1))
    self.first_layer = residual_block[0]
    self.layers = torch.nn.Sequential(*residual_block[1:])
    self.activation = torch.nn.ReLU()

  def forward(self, x):
    s = self.first_layer(x)
    x = self.layers(x)
    x = s + x
    return self.activation(x)

class SpeechRecognitionModel(torch.nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.layers = torch.nn.Sequential(
      ResidualBlock(1, 16, 2),
      ResidualBlock(16, 32, 2),
      ResidualBlock(32, 64, 3),
      ResidualBlock(64, 128, 3),
      ResidualBlock(128, 128, 3),
      torch.nn.AvgPool1d(kernel_size=3, stride=3),
      torch.nn.Flatten(),
    )
    self.layers2 = torch.nn.Sequential(
      torch.nn.Linear(682624, 256),
      torch.nn.ReLU(),
      torch.nn.Linear(256, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, num_classes),
      torch.nn.ReLU(),
      torch.nn.Softmax(dim=1)
    )

  def forward(self, x):
    print(x.shape, flush=True)
    x = self.layers(x)
    print(x.shape, flush=True)
    x = self.layers2(x)
    return x