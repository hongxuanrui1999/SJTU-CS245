import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, models, transforms

model = models.resnet152(pretrained=False)

# torch.save(model.state_dict(), './model_res152.pkl')
model.load_state_dict(torch.load('./model_res152.pkl'))