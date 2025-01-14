import os
import torch
import numpy as np
import subprocess as sp
from typing import Tuple
from torch_geometric.utils import to_dense_adj


def byte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(value / (1024 * 1024), digits)

def medibyte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(1.0485 * value, digits)

def get_gpu_memory_from_nvidia_smi(  # pragma: no cover
    device: int = 0,
    digits: int = 2,
) -> Tuple[float, float]:
    r"""Returns the free and used GPU memory in megabytes, as reported by
    :obj:`nivdia-smi`.

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

    free_mem = medibyte_to_megabyte(int(free_out[device].split()[0]), digits)
    used_mem = medibyte_to_megabyte(int(used_out[device].split()[0]), digits)

    return free_mem, used_mem

def find_devices(max_devices: int = 1, greedy: bool = False, gamma: int = 12):
    # if no gpus are available return None
    if not torch.cuda.is_available():
        return max_devices
    n_gpus = torch.cuda.device_count()
    # if only 1 gpu, return 1 (i.e., the number of devices)
    if n_gpus == 1:
        return 1
    # if multiple gpus are available, return gpu id list with length max_devices
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices is not None:
        visible_devices = [int(i) for i in visible_devices.split(',')]
    else:
        visible_devices = range(n_gpus)
    available_memory = np.asarray([get_gpu_memory_from_nvidia_smi(device)[0]
                                   for device in visible_devices])
    # if greedy, return `max_devices` gpus sorted by available capacity
    if greedy:
        devices = np.argsort(available_memory)[::-1].tolist()
        return devices[:max_devices]
    # otherwise sample `max_devices` gpus according to available capacity
    p = (available_memory / np.linalg.norm(available_memory, gamma)) ** gamma
    # ensure p sums to 1
    p = p / p.sum()
    devices = np.random.choice(np.arange(len(p)), size=max_devices,
                               replace=False, p=p)
    return devices.tolist()

def reduce_precision():
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class TensorsCache:
    """
    Compute and cache large tensor (adj, degree, positive weights) 
    that are used multiple times in the forward pass.
    """    
    def __init__(self, use_cache=True):
        if use_cache:
            self.__cache_dict__ = {}
        else:
            self.__cache_dict__ = None

    def __get_tensors(self, key, build_val):
        if self.__cache_dict__ is None:
            return build_val()
        else:
            if key not in self.__cache_dict__:
                v = build_val()
                self.__cache_dict__[key] = v
            return self.__cache_dict__[key]

    def get_and_cache_A(self, edge_index=None, batch=None):
        return self.__get_tensors('adj', lambda: to_dense_adj(edge_index, batch))

    def get_and_cache_pos_weight(self, adj=None, node_mask=None):

        def __compute_pos_weight():
            if node_mask is not None:
                N = node_mask.sum(-1).view(-1, 1, 1)  # has shape B x 1 x 1
            else:
                N = adj.shape[-1]

            # the clamp is needed to avoid zero division when we have all edges
            n_edges = torch.clamp(adj.sum([-1, -2]), min=1).view(-1, 1, 1)  # this is a vector of size B x 1 x 1
            n_not_edges = torch.clamp(N**2 - n_edges, min=1).view(-1, 1, 1)   # this is a vector of size B x 1 x 1
            
            return (N ** 2 / n_edges) * adj + (N ** 2 / n_not_edges) * (1 - adj)

        return self.__get_tensors('pos_weight', __compute_pos_weight)

    def get_and_cache_D(self, adj):
        return self.__get_tensors('D', lambda: TensorsCache._rank3_diag(torch.einsum('ijk->ij', adj)))

    @staticmethod
    def _rank3_diag(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(x.size(1)).type_as(x)
        return eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))