"""## Setup"""

import wfdb
import math
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""## Utils"""

path='D:\Jasiu\Dokumenty\studia\inzynierka\project\qt-database-1.0.0\\'
records=glob.glob(path+"*.dat")

def mapChartSectionToColor(features):
        if features['P']: return 'red'
        if features['QR']: return 'green'
        if features['RS']: return 'blue'
        if features['ST']: return 'orange'
        if features['T']: return 'yellow'

def plotecg(signal,features):
    preview_start = 0
    preview_end = 600
    fig = wfdb.plot_items(signal=signal[preview_start:preview_end], fs=250, title='', return_fig=True)
    temp = preview_start
    for i in range(preview_start,preview_end):
        try:
            chartColor = mapChartSectionToColor(features.loc[i, :])
            plt.axvspan(temp,i, color=chartColor, alpha=0.1)
            temp = i
        except:
            pass
    plt.legend()
    plt.show()

def SymbolsOneHotEncoding(path): 
    extention = 'q1c'
    annotation = wfdb.rdann(path, extension=extention)
    sampfrom=min(annotation.sample)
    sampto=max(annotation.sample)

    signals, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto, channels=[0])

    annotation = wfdb.rdann(path, extention, sampfrom=sampfrom, sampto=sampto)
   # print(fields['fs'])
    d = {'symbol': annotation.symbol}
    df = pd.DataFrame(d, (annotation.sample-sampfrom))
    sym = df['symbol']
    df['P'] = 0
    df['QR'] = 0
    df['RS'] = 0
    df['ST'] = 0
    df['T'] = 0
    
    for i in range(1, len(df)):
        if sym.iloc[i-1] == "(" and sym.iloc[i] == "p" and sym.iloc[i+1] == ")": df.iloc[i-1:i+2, 1] = 1
        if sym.iloc[i-1] == "(" and sym.iloc[i] == "N": df.iloc[i-1:i+1, 2] = 1
        if sym.iloc[i-1] == "N" and sym.iloc[i] == ")": df.iloc[i:i+1, 3] = 1
        if sym.iloc[i-1] == "N" and sym.iloc[i] == ")" and sym.iloc[i+1] == "(": df.iloc[i+1:i+2, 4] = 1
        if sym.iloc[i-1] == "(" and sym.iloc[i] == "t" and sym.iloc[i+1] == ")": df.iloc[i:i+2, 5] = 1

    #print(df.head(15))

    return signals, df

def Segmentation(records):
    data = []
    for rec in records:
        path = rec[:-4]
        x,y=SymbolsOneHotEncoding(path)
        data.append((x,y))
    return data

"""## Main"""

data=Segmentation(records[:20]) 
#print(data[0][1])
# for i in range(0, len(data)):
#     plotecg(data[i][0],data[i][1])
#     plt.close()

# x = list()
# for i in range(len(data)):
#     x.append(data[i][0])
# y = list()
# for i in range(len(data)):
#     y.append(data[i][1].reset_index())
      
def fragmentation(data):
    P = []
    QRS=[]
    T=[]
    booler =  False
    for i in data:
        signal = i[0]
        features = i[1].reset_index()
        for  index, symbol in zip(features['index'], features['symbol']):
            if booler ==True:
                stop = index 
                booler =False
                if previousSymbol == 'p':
                    P.append(signal[start:stop])
                if previousSymbol == 'N':
                    
                    QRS.append(signal[start:stop])
                if previousSymbol == 't':
                    T.append(signal[start:stop])
                    
            if symbol == 'p':
                start = previousIndex
                booler =True
            if symbol == 'N':
                start = previousIndex
                booler =True
            if symbol == 't':
                start = previousIndex
                booler =True
            previousIndex=index
            previousSymbol = symbol

    return [P,QRS,T]

waves= fragmentation(data)
images(waves)

def images(waves):   
    path='D:\Jasiu\Dokumenty\studia\inzynierka\project\images\'     
    for i in waves[0]:
        fig = plt.figure(
        plt.plot(i)
        plt.savefig(path+'P\{}'.format(i))
        plt.close(fig)
    for i in waves[1]:
        fig = plt.figure(
        plt.plot(i)
        plt.savefig(path+'QRS\{}'.format(i))
        plt.close(fig)
    for i in waves[2]:
        fig = plt.figure(
        plt.plot(i)
        plt.savefig(path+'T\{}'.format(i))
        plt.close(fig)