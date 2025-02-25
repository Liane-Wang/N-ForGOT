import torch
import torch.nn as nn
from typing import List

from . import Finetune, NForGOT, Joint

import importlib

module_list = {
    'Finetune': Finetune,
    'Joint': Joint,
    'NForGOT':NForGOT,
}

def get_model(args, neighbor_finder, node_features, edge_features, src_label, dst_label):
    try:
        method = getattr(module_list[args.method], args.method)
        return method(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
    except ImportError:
        print(f"Method '{args.method}' not found.")
    return None