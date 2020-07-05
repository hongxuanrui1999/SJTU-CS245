import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn, optim
import warnings
import time

model = models.resnet152(pretrained=True)

torch.save(model.state_dict(), './model_res152.pkl')
# model.load_state_dict(torch.load('./model_res152.pkl'))