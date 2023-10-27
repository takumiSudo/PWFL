from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import flwr as fl
from flwr.server.client_manager import ClientManager

from data import *
from util import *
from client import *
from strat import *
from server import Server
from pServer import pServer
from zeroServer import zServer
from clientmanager import SimpleClientManager

if __name__ == "__main__":
    
    client_resources  = None

    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        server = pServer(client_manager=SimpleClientManager()),
        config=fl.server.ServerConfig(num_rounds=1),
        client_resources=client_resources,
        # strategy = FedAvg(),
    )