from typing import Mapping, Optional, Union
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger


class CustomTensorBoardLogger(TensorBoardLogger):
    """Extended TensorBoardLogger with additional logging capabilities"""
    
    def __init__(self,
                 save_dir: str,
                 name: Optional[str] = "default",
                 version: Optional[Union[int, str]] = None,
                 **kwargs):
        super().__init__(save_dir=save_dir, name=name, version=version, **kwargs)
        
    def log_figure(self, fig, name: str = 'figure', close: bool = True, global_step: int = None):
        """Log a matplotlib figure"""
        if global_step is None:
            raise ValueError("global_step must be provided")
        self.experiment.add_figure(name, fig, global_step=global_step)
        if close:
            plt.close(fig)
            
    def log_nx_graph_plot(self, adj, signal: Optional[np.ndarray] = None,
                         with_labels: Optional[bool] = False,
                         node_size=25, font_size=12,
                         name: str = 'graph', pos: Optional[np.ndarray] = None,
                         labels: Optional[Mapping] = None,
                         cmap=None, vmax: Optional[float] = None,
                         vmin: Optional[float] = None,
                         global_step: Optional[int] = None):
        """Log a networkx graph visualization"""
        cmap = cmap or plt.cm.viridis
        vmax = vmax or (np.max(signal) if signal is not None else None)
        vmin = vmin or (np.min(signal) if signal is not None else None)

        if hasattr(adj, 'toarray'):
            graph = nx.from_scipy_sparse_array(adj)
        else:
            graph = nx.from_numpy_array(adj)
            
        if pos is None:
            pos = nx.kamada_kawai_layout(graph)
            
        fig, ax = plt.subplots()
        
        if signal is not None:
            if hasattr(signal, 'dtype') and signal.dtype == int:
                nx.draw(graph, node_color=signal, ax=ax, node_size=node_size,
                       pos=pos, with_labels=with_labels, labels=labels,
                       font_size=font_size)
            elif hasattr(signal, 'dtype'):
                nx.draw(graph, node_color=signal, cmap=cmap, vmax=vmax,
                       vmin=vmin, ax=ax, node_size=node_size, pos=pos,
                       with_labels=with_labels, labels=labels,
                       font_size=font_size)
            elif signal == 'white':
                nx.draw(graph, node_color=signal, ax=ax, node_size=50,
                       pos=pos, with_labels=with_labels, labels=labels)
        else:
            nx.draw(graph, ax=ax, node_size=25, pos=pos,
                   with_labels=with_labels, labels=labels)
            
        self.log_figure(fig, name, global_step=global_step)

    def finalize(self, status: str):
        """Clean up logging"""
        self.save() 