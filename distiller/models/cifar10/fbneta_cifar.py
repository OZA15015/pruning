import torch                                                                                                                            
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import torchvision.datasets as datasets
import sys

#from general_functions.dataloaders import get_loaders, get_test_loader
#from general_functions.utils import get_logger, weights_init, create_directories_from_list

#import distiller.models.cifar10.fbnet_building_blocks.fbnet_builder as fbnet_builder

sys.path.append('/home/oza/pre-experiment/FBNet')
import fbnet_building_blocks.fbnet_builder as fbnet_builder
#from architecture_functions.training_functions import TrainerArch
#from architecture_functions.config_for_arch import CONFIG_ARCH
#from collections import OrderedDict
#import torchvision
#import torchvision.transforms as transforms

__all__ = ['fbneta_cifar']

def fbneta_cifar():
    model = fbnet_builder.get_model('fbnet_a', cnt_classes=10).cuda()
    return model
