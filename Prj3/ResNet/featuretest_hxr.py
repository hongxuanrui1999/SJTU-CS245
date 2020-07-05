import numpy as np
import torch
import torchvision
import os
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import nn, optim
from SelectiveSearch import SelectiveSearchImg
from multiprocessing import Process, Queue, freeze_support

IMG_PATH = '../AwA2-data/JPEGImages/'
LD_PATH = '../AwA2-data/DL_LD/'

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

data_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model = models.resnet152(pretrained=False)
model.load_state_dict(torch.load('./model_res152.pkl'))
exact_list = ['avgpool']
exactor = FeatureExtractor(model, exact_list)

def extract(className, imgName):
    props = SelectiveSearchImg(className, imgName)
    feature_list = []
    features = None
    for img in props:
        with torch.no_grad():
            img = data_tf(img)
        img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        feature = exactor(img)[0]
        feature = feature.resize(feature.shape[0], feature.shape[1])
        feature_list.append(feature.detach().cpu().numpy())
        features = np.row_stack(feature_list)
    if len(feature_list) == 0:
        print("Fatal error at %s" % (imgName))
    return features

class FEProcess(Process):
    def __init__(self, class_dict):
        super(FEProcess, self).__init__()
        self.class_dict = class_dict
    
    def run(self):
        for className, totalNum in self.class_dict.items():
            print("SS at %s" % (className))
            for idx in range(10001, totalNum + 1):
                if not os.path.exists(LD_PATH + className + '/' + className + '_' + str(idx) + '.npy'):
                    des = extract(className, className + '_' + str(idx))
                    np.save(LD_PATH + className + '/' + className + '_' + str(idx), des)
    
def main():
    dict_list = np.load('../f_class_dict_2.npy', allow_pickle=True)
    freeze_support()
    processPool = [FEProcess(dict_list[i]) for i in range(8)]
    for i in range(8):
        processPool[i].start()
    for i in range(8):
        processPool[i].join()

if __name__ == '__main__':
    main()
