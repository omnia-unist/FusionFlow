import torchvision.transforms as transforms

if __debug__:
    import logging  
    import time
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

class FullTraceCompose(transforms.Compose):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.
        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)
        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.
    """

    def __call__(self, img):
        for t in self.transforms:
            start=time.perf_counter()
            img = t(img)
            end=time.perf_counter()
            log.debug(f"Transform {t} END at_time {end-start}")
                
        return img
