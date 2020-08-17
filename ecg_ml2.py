"""## Setup"""
import wfdb
import glob
import matplotlib.pyplot as plt
import pandas as pd

"""## Utils"""

path1='D:\Jasiu\Dokumenty\studia\inzynierka\project\qt-database-1.0.0\\'
records=glob.glob(path1+"*.dat")


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
    # df['P'] = 0
    # df['QR'] = 0
    # df['RS'] = 0
    # df['ST'] = 0
    # df['T'] = 0
    
    # for i in range(1, len(df)):
    #     if sym.iloc[i-1] == "(" and sym.iloc[i] == "p" and sym.iloc[i+1] == ")": df.iloc[i-1:i+2, 1] = 1
    #     if sym.iloc[i-1] == "(" and sym.iloc[i] == "N": df.iloc[i-1:i+1, 2] = 1
    #     if sym.iloc[i-1] == "N" and sym.iloc[i] == ")": df.iloc[i:i+1, 3] = 1
    #     if sym.iloc[i-1] == "N" and sym.iloc[i] == ")" and sym.iloc[i+1] == "(": df.iloc[i+1:i+2, 4] = 1
    #     if sym.iloc[i-1] == "(" and sym.iloc[i] == "t" and sym.iloc[i+1] == ")": df.iloc[i:i+2, 5] = 1

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


data=Segmentation(records[:]) 

#częstość akcji serc, amplitudę zespołu QRS, oraz czasy trwania głównych załamków P i T
#oraz zespołu QRS w dowolnym sygnale EKG.

def Labels(signals):
    data = []
    booler1 = False
    booler2 = False
    booler3 = False
    
    for i in signals:
        features = i[1].reset_index()
        signal = i[0]
        Rpeaks=[]
        QRSamplitudes=[]
        Plength=[]
        Tlength=[]
        QRSlength=[]
        labels = pd.DataFrame(columns=['heart rate', 'QRS amplitude', 'P length', 'T length', 'QRS length'])
    
        for inde, index, symbol in zip(features.index, features['index'], features['symbol']):
            
            if (symbol == 'N'):
                Rpeaks.append(index)
                QRSamplitudes.append(signal[index])
                
            if (booler1):
                stop = index
                Plength.append(stop-start)
                booler1 = False 
            if (symbol == 'p' ):
                start = previousIndex
                booler1 =True
                
            if (booler2):
                stop = index
                Tlength.append(stop-start)
                booler2 = False 
            if (symbol == 't' ):
                start = previousIndex
                booler2 =True
                
            if (booler3):
                stop = index
                QRSlength.append(stop-start)
                booler3 = False 
            if (symbol == 'N' ):
                start = previousIndex
                booler3 =True

                
            previousIndex = index
            
            
        if(len(Rpeaks) != 0):
            labels.at[0,'heart rate']=sum(Rpeaks)/len(Rpeaks)
        if(len(QRSamplitudes) != 0):
            labels.at[0,'QRS amplitude']=float(sum(QRSamplitudes)/len(QRSamplitudes))
        if(len(Plength) != 0):
            labels.at[0,'P length']=sum(Plength)/len(Plength)
        if(len(Plength) != 0):
            labels.at[0,'T length']=sum(Tlength)/len(Tlength)
        if(len(QRSlength) != 0):
            labels.at[0,'QRS length']=sum(QRSlength)/len(QRSlength)
            data.append((i[0],labels))
             
    return data
        

data2 = Labels(data)
    
    
    
    
    
    