import os
import torch
import numpy as np
import subprocess as sp
from typing import Tuple


def _mebibyte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(1.0485 * value, digits)


def _get_gpu_memory_from_nvidia_smi(  # pragma: no cover
    device: int = 0,
    digits: int = 2,
) -> Tuple[float, float]:
    r"""Returns the free and used GPU memory in megabytes, as reported by
    :obj:`nvidia-smi`.

    .. note::

        :obj:`nvidia-smi` will generally overestimate the amount of memory used
        by the actual program, see `here <https://pytorch.org/docs/stable/
        notes/faq.html#my-gpu-memory-isn-t-freed-properly>`__.

    Args:
        device (int, optional): The GPU device identifier. (default: :obj:`1`)
        digits (int): The number of decimals to use for megabytes.
            (default: :obj:`2`)
    """
    CMD = 'nvidia-smi --query-gpu=memory.free --format=csv'
    free_out = sp.check_output(CMD.split()).decode('utf-8').split('\n')[1:-1]

    CMD = 'nvidia-smi --query-gpu=memory.used --format=csv'
    used_out = sp.check_output(CMD.split()).decode('utf-8').split('\n')[1:-1]

    if device < 0 or device >= len(free_out):
        raise AttributeError(
            f'GPU {device} not available (found {len(free_out)} GPUs)')

    free_mem = _mebibyte_to_megabyte(int(free_out[device].split()[0]), digits)
    used_mem = _mebibyte_to_megabyte(int(used_out[device].split()[0]), digits)

    return free_mem, used_mem

def find_devices(max_devices: int = 1, greedy: bool = False, gamma: int = 12):
    if not torch.cuda.is_available():
        return max_devices
    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        return 1
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices is not None:
        visible_devices = [int(i) for i in visible_devices.split(',')]
    else:
        visible_devices = range(n_gpus)

    available_memory = np.asarray([_get_gpu_memory_from_nvidia_smi(device)[0]
                                   for device in visible_devices])
    if greedy:
        idx_to_sort = np.argsort(available_memory)[::-1].tolist()
        idx_devices = idx_to_sort[:max_devices]
    else:
        p = (available_memory / np.linalg.norm(available_memory, gamma)) ** gamma
        p = p / p.sum()
        idx_devices = np.random.choice(np.arange(len(p)), size=max_devices,
                                       replace=False, p=p)

    return [visible_devices[i] for i in idx_devices]

def reduce_precision():
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
