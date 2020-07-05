import pandas as pd
import numpy as np

PATH = '../Office-Home_resnet50/'

def loadData(sourceName, targetName):
    '''
        Use to load source domain and target domain data from .csv file\n
    '''
    colName = [i for i in range(2048)]
    colName.append('label')
    source = pd.read_csv(PATH + sourceName + '_' + sourceName + '.csv', names=colName)
    target = pd.read_csv(PATH + sourceName + '_' + targetName + '.csv', names=colName)
    sourceData = source[colName[:-1]].values
    sourceLabel = source['label'].values.astype(int)
    targetData = target[colName[:-1]].values
    targetLabel = target['label'].values.astype(int)
    print('Loaded Source domain: %s in shape' % (sourceName), sourceData.shape)
    print('Loaded Target domain: %s in shape' % (targetName), targetData.shape)
    return sourceData, sourceLabel, targetData, targetLabel