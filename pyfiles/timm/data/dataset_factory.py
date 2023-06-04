import os

from .dataset import IterableImageDataset, ImageDataset, FullTraceCopyedDistMemDataset, CopyedDistMemDataset, SyntheticDataset
from torchvision.datasets import ImageFolder 

def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == 'validation':
        try_root = os.path.join(root, 'val')
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(name, root, split='validation', search_split=True, is_training=False, batch_size=None, **kwargs):
    name = name.lower()
    if name.startswith('tfds'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    elif name.startswith('native'):
        ds = ImageFolder(root, **kwargs)
    elif name.find("fulltrace") != -1:
        ds = FullTraceCopyedDistMemDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    elif name.startswith("prep"):
        ds = CopyedDistMemDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    elif name.startswith("load") | name.startswith("train"):
        ds = SyntheticDataset(
            root, parser=name, split=split, is_training=is_training, batch_size=batch_size, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop('repeats', 0)  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds
