r"""Utility classes & functions for data loading. Code in this folder is mostly
used by ../dataloder.py.

A lot of multiprocessing is used in data loading, which only supports running
functions defined in global environment (py2 can't serialize static methods).
Therefore, for code tidiness we put these functions into different files in this
folder.
"""

import sys
import atexit

# old private location of the ExceptionWrapper that some users rely on:
from torch._utils import ExceptionWrapper


IS_WINDOWS = sys.platform == "win32"


MP_STATUS_CHECK_INTERVAL = 5.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""


python_exit_status = False
r"""Whether Python is shutting down. This flag is guaranteed to be set before
the Python core library resources are freed, but Python may already be exiting
for some time when this is set.

Hook to set this flag is `_set_python_exit_flag`, and is inspired by a similar
hook in Python 3.7 multiprocessing library:
https://github.com/python/cpython/blob/d4d60134b29290049e28df54f23493de4f1824b6/Lib/multiprocessing/util.py#L277-L327
"""


def _set_python_exit_flag():
    global python_exit_status
    python_exit_status = True

atexit.register(_set_python_exit_flag)

REMOTE_PREP_PIPE = -11 # "This is Pipeline 1 in case when Decode is in read part"
REMOTE_DECODE_PREP_PIPE = -22   # This is Pipeline 1. Due to Prep beeing Decode + Augment, it became like this. In case of decode being in read, it is "Pipe2"
REMOTE_DECODE_PREP_COLLATE_PIPE = -33 # "This is Pipeline 3 in case when Decode is in read part"
REMOTE_READ_PREP_WITH_DECODE_PIPE = -55  # This is Pipeline 2 when Decode is in Prep Part. Requires DB on the remote side
REMOTE_READ_PREP_WITH_DECODE_COLLATE_PIPE = -66  # This is Pipeline 3 when Decode is in Prep Part. Required DB on the remote side
COLLATE_ONLY = -11111

from . import worker, signal_handling, pin_memory, collate, fetch
