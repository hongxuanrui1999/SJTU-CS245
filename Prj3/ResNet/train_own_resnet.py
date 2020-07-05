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

max_epoch=1
lr_init = 0.001

start_time = time.time()

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

path = 'Animals_with_Attributes2/'

classname = pd.read_csv(path + 'classes.txt', header=None, sep='\t')
dic_class2name = {classname.index[i]: classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]: classname.index[i] for i in range(classname.shape[0])}


class dataset(Dataset):
    def __init__(self, data, label, transform):
        super().__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return self.data.shape[0]


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


data = np.load(path + 'AWA2_224_traindata.npy')
label = np.load(path + 'AWA2_trainlabel.npy')

data_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = dataset(data, label, data_tf)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = models.resnet152(pretrained=True)
num_fc_ftr = model.fc.in_features
model.fc = torch.nn.Linear(num_fc_ftr, 50)

criterion = nn.CrossEntropyLoss()                                                  
optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)    

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

for epoch in range(max_epoch):
    print(epoch)
    loss_sigma = 0.0   
    correct = 0.0
    total = 0.0
    scheduler.step() 
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()

        if i % 1 == 0:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(loader), loss_avg, correct / total))
            print(time.time()-start_time)

# torch.save(model, path + 'AWA2_224_traindata.npy')
# torch.save(model.state_dict(), './model.pkl')
