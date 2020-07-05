"""## Setup"""

import wfdb
import math
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
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

data=Segmentation(records[:41]) 
    
def smooth(x,window_len=11,window='hanning'):
    x = x.T[0]
    if x.ndim != 1:
        raise ValueError #"smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError #"Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError # "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

for i in data:
    i = (smooth(i[0]), i[1])

for i in range(0, len(data)):
    plotecg(data[i][0],data[i][1])
    plt.close()
    
def fragmentation(data):
    signals = []
    booler = True
    stop=0
    start=0
    indeStop=0
    indeStart=0
    
    for i in data:
         signal = i[0]
         features = i[1].reset_index()
         for  inde, index, symbol in zip(features.index, features['index'], features['symbol']):
             
             if (symbol == 'p' and booler == False):
                 indeStop = previousInde
                 stop = previousIndex
                 if(start<stop):
                     signals.append((signal[start:stop],features[indeStart:indeStop]))
                 booler=True
             if (symbol == 'p' and booler==True):
                 start = previousIndex
                 indeStart = previousInde
                 booler =False
             
             previousInde=inde
             previousIndex=index
             previousSymbol = symbol

    return signals        

signals=fragmentation(data)

for i in range(len(signals)-1, 0 , -1):
    if signals[i][0].shape[0] >2000:
        signals.pop(i)
        
def EcgToImages(signals,path):
    y=0
    for i in signals:
       fig = plt.figure()
       plt.ylim(min(i[0]), max(i[0]))
       plt.xlim(0, signals[y][0].shape[0])
       plt.plot(i[0])
       plt.axis('off')
       fig.savefig(path+'\{}.png'.format(y))
       plt.close(fig)       
       y=y+1

path='D:\Jasiu\Dokumenty\studia\inzynierka\project\images'
# EcgToImages(signals,path)

def readImages():
    img = []
    images=glob.glob(path+"\*.png")
    for i in images: 
        img.append(mpimg.imread(i))
    return img

images = readImages()

