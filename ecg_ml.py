"""## Setup"""
import wfdb
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
import sys
import xml.etree.ElementTree
from pascal_voc_writer import Writer

"""## Utils"""

path1='D:\Jasiu\Dokumenty\studia\inzynierka\project\qt-database-1.0.0\\'
records=glob.glob(path1+"*.dat")

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

for i in range(0, 5):
    plotecg(data[i][0],data[i][1])
    plt.close()

def fragmentation(data):
    signals = []
    booler = True
    stop = 0
    start = 0
    indeStop = 0
    indeStart = 0

    for i in data:
        signal = i[0]
        features = i[1].reset_index()
        for inde, index, symbol in zip(features.index, features['index'], features['symbol']):
  
            if (symbol == 'p' and booler == False):
                indeStop = previousInde
                stop = previousIndex
                
                if (start < stop):
                    annotation=features[indeStart:indeStop]
                    
                    for i in annotation['index']: 
                        
                        x = i-start
                        annotation['index'].replace(i,x,inplace=True)
                        
                    signals.append((signal[start:stop], annotation))
                    
                booler = True
            if (symbol == 'p' and booler == True):
                start = previousIndex
                indeStart = previousInde
    
                booler = False
    
            previousInde = inde
            previousIndex = index
            previousSymbol = symbol
                

    return signals

signals = fragmentation(data[:])

for i in range(len(signals)-1, 0 , -1):
    if signals[i][0].shape[0] >2000:
        signals.pop(i)

def EcgToImages():
    y = 0
    
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    app.quit()

    for i in signals:
        fig = plt.figure(figsize=(90/dpi,60/dpi),dpi=dpi)
        plt.ylim(min(i[0]), max(i[0]))
        plt.xlim(0, signals[y][0].shape[0])
        plt.plot(i[0])
        plt.axis('off')
        fig.savefig('D:\Jasiu\Dokumenty\studia\inzynierka\project\data\{}.png'.format(y))
        if y%3==0:
            fig.savefig('D:\Jasiu\Dokumenty\studia\inzynierka\project\dat\\validation\images\{}.png'.format(y))
        else:
            fig.savefig('D:\Jasiu\Dokumenty\studia\inzynierka\project\dat\\train\images\{}.png'.format(y))
        plt.close(fig)
        y = y + 1

def readImages():
    images = glob.glob(path2 + '\*.png')
    for i in range(len(images)):
        yield cv2.imread(path2+'\{}.png'.format(i))

path2='D:\Jasiu\Dokumenty\studia\inzynierka\project\data'
EcgToImages()
images = readImages()



def createBoxes(signals, img):
    a=0
    for i in signals:
        
        img = next(images)
        img2 = img[:, :, 0].reshape([img.shape[0], img.shape[1]])
        for j in range(img.shape[1]):
            if min(img2[:, img.shape[1] - 1 - j]) < 255:
                max_right = img.shape[1] - j
                break
        for j in range(img.shape[1]):
            if min(img2[:, j])< 255:
                max_left = j-1
                break
            
        for j in range(img.shape[0]):
            if min(img2[j, : ]) < 255:
                max_top = j - 1
                break
        for j in range(img.shape[0]):
            if min(img2[img.shape[0]-1-j, : ])< 255:
                max_bot = img.shape[0]-j
                break
        max_x_pixels = max_right - max_left
        max_y_pixels = max_bot - max_top
        
        signal_max_x = i[0].shape[0]
        signal_max_y = abs(max(i[0]) - min(i[0]))
        signal_max_y = signal_max_y[0]
        
        scale_X = max_x_pixels / signal_max_x
        scale_Y = max_y_pixels / signal_max_y
        
        y = 0
        
        features = i[1] 
        boolerP=False
        boolerN=False
        boolerT=False
        for index, symbol in zip(features['index'], features['symbol']):
            if boolerP == True:
                x_wave_endP = index
                boolerP=False
            if symbol == 'p':
                x_wave_beginP = previousIndex
                boolerP=True
            
            if boolerN == True:
                x_wave_endN= index
                boolerN=False
            if symbol == 'N':
                x_wave_beginN = previousIndex
                boolerN=True
            
            if boolerT == True:
                x_wave_endT= index
                boolerT=False
            if symbol == 't':
                x_wave_beginT = previousIndex
                boolerT=True
                
            previousIndex = index
        
        y_wave_botP = min(i[0][x_wave_beginP:x_wave_endP])[0] - min(i[0])[0]
        y_wave_topP = max(i[0][x_wave_beginP:x_wave_endP])[0] - min(i[0])[0]
    
        y_wave_botN = min(i[0][x_wave_beginN:x_wave_endN])[0] - min(i[0])[0]
        y_wave_topN = max(i[0][x_wave_beginN:x_wave_endN])[0] - min(i[0])[0]
        
        y_wave_botT = min(i[0][x_wave_beginT:x_wave_endT])[0] - min(i[0])[0]
        y_wave_topT = max(i[0][x_wave_beginT:x_wave_endT])[0] - min(i[0])[0]
        
        pt1P = (max_left + int(x_wave_endP * scale_X), max_bot - int(y_wave_topP * scale_Y)-2)
        pt2P = (max_left + int(x_wave_beginP * scale_X), max_bot - int(y_wave_botP* scale_Y))
        
        pt1N = (max_left + int(x_wave_endN * scale_X), max_bot - int(y_wave_topN * scale_Y)-2)
        pt2N = (max_left + int(x_wave_beginN * scale_X), max_bot - int(y_wave_botN* scale_Y))
    
        pt1T = (max_left + int(x_wave_endT * scale_X), max_bot -  int(y_wave_topT * scale_Y)-2)
        pt2T = (max_left + int(x_wave_beginT * scale_X), max_bot -  int(y_wave_botT* scale_Y))
        
  
        
        writer = Writer('D:\Jasiu\Dokumenty\studia\inzynierka\project\data\images\{}.png'.format(a), 640, 480)
        
        writer.addObject('P', pt2P[0], pt1P[1], pt1P[0], pt2P[1])
        writer.addObject('QRS', pt2N[0], pt1N[1], pt1N[0], pt2N[1])
        writer.addObject('T', pt2T[0], pt1T[1], pt1T[0], pt2T[1])
       
        if a%3==0:
            writer.save('D:\Jasiu\Dokumenty\studia\inzynierka\project\dat\\validation\\annotations\{}.xml'.format(a))
        else:
            writer.save('D:\Jasiu\Dokumenty\studia\inzynierka\project\dat\\train\\annotations\{}.xml'.format(a))
        a=a+1
        
        # img_box = cv2.rectangle(img, pt1P, pt2P, (255, 30, 30), thickness=1)
        # img_box = cv2.rectangle(img, pt1N, pt2N, (255, 30, 30), thickness=1)
        # img_box = cv2.rectangle(img, pt1T, pt2T, (255, 30, 30), thickness=1)
        # fig = plt.figure()

        # plt.imshow(img_box)
        # plt.axis('off')
        # plt.show()
        # plt.close()
        

createBoxes(signals[:],images)


from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="D:\Jasiu\Dokumenty\studia\inzynierka\project\dat")
trainer.setTrainConfig(object_names_array=["P","QRS","T"], batch_size=2, num_experiments=50,train_from_pretrained_model="D:\Jasiu\Dokumenty\studia\inzynierka\project\pretrained-yolov3.h5")
trainer.trainModel()

# import tensorflow as tf
# tf.test.is_gpu_available() # True/False

# # Or only check for gpu's with cuda support
# tf.test.is_gpu_available(cuda_only=True) 

