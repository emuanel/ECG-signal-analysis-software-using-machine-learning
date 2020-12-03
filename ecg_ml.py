"""## Setup"""
import wfdb
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
import sys
from pascal_voc_writer import Writer
import xml.dom.minidom as md 
from imageai.Detection.Custom import DetectionModelTrainer

"""## Utils"""

def ReadSignals(path):
    """
    Read signals with annotations from files
    :param str path: paths to records
    :return: list path: paths to records
             DataFrame df: annotations
    """
    extention = 'q1c'
    annotation = wfdb.rdann(path, extension=extention)
    sampfrom=min(annotation.sample)
    sampto=max(annotation.sample)
    signals, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto, channels=[0,1])
    annotation = wfdb.rdann(path, extention, sampfrom=sampfrom, sampto=sampto)
    d = {'symbol': annotation.symbol}
    df = pd.DataFrame(d, (annotation.sample-sampfrom))
    return signals, df

def Segmentation(records):
    """
    Make list of tuples containing signal with annotations
    :param list records: paths to records in directory
    :return: list data: tuples containing signal and annotations
    """
    data = []
    for rec in records:
        path = rec[:-4]
        x,y=ReadSignals(path)
        data.append((x[:,0],y))
        data.append((x[:,1],y))
    return data

path1='qt-database-1.0.0\\'
records=glob.glob(path1+"*.dat")
data=Segmentation(records[:]) 

def Smooth(x, window_len=11, window='hanning'):
    """
    smoothing the signal - noise reduction
    :param array x: signal
    :param int window_len: 
    :param str window: type of window function
    :return: array y: smoothed signal
    """
    
    if x.ndim != 1:
        raise ValueError #"smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError #"Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError # "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

for i in data:
    i = (Smooth(i[0]), i[1])
    
def Fragmentation(data):
    """    
    splitting the signals into sections from P wave to next P wave
    :param list data: containing tuples with signal and annotation
    :return: list data: containing more tuples with signal, annotation and origin information
    """
    signals = [] #list with new data
    beginning = True #boolean variable to check if index should be the beginning of the signal or the end
    start = 0 #start of new signal
    indeStop = 0 #end of new annotation
    indeStart = 0 #start of new annotation
    previousInde = 0 #index in dataframe with annotations from previous iteration
    previousIndex = 0 #index in array with signal from previous iteration
    
    for i,a in zip(data, range(len(data))):
        signal = i[0]
        features = i[1].reset_index() 
        
        for inde, index, symbol in zip(features.index, features['index'], features['symbol']): #iteration over three columns
            if (symbol == 'p' and beginning == False):
                indeStop = previousInde
                stop = previousIndex #stop of new signal
                
                if (start < stop):
                    annotation=features[indeStart:indeStop]
                    
                    for i in annotation['index']:
                        x = i-start
                        annotation['index'].replace(i,x,inplace=True)
                    signals.append((signal[start:stop], annotation, a)) #add new signal with annotation
                beginning = True
                
            if (symbol == 'p' and beginning == True):
                start = previousIndex
                indeStart = previousInde
                beginning = False
    
            previousInde = inde
            previousIndex = index
    return signals

signals = Fragmentation(data[:])

for i in range(len(signals)-1, 0 , -1):
    if signals[i][0].shape[0] >500:
        signals.pop(i)

def EcgToImages():
    """
    Transform signals 1d to images 2d and save them to appropriate folders
    """
    
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()  #read screen dpi (dots per inch)
    app.quit()
    
    y = 0 #iterator
    for i in signals:  
        max=np.amax(data[i[2]][0])
        min=np.amin(data[i[2]][0])
        fig = plt.figure(figsize=(640/dpi,480/dpi),dpi=dpi)
        plt.ylim(min, max)
        plt.xlim(0, 500)
        plt.plot(i[0])  #convert 1d signal to 2d signal
        plt.axis('off')
        fig.savefig('data\{}.png'.format(y))
        if y%6==0:
            fig.savefig('dat\\test\images\{}.png'.format(y))
            plt.close(fig)
            y = y + 1
            continue
        if y%3==0:
            fig.savefig('dat\\validation\images\{}.png'.format(y))
        else:
            fig.savefig('dat\\train\images\{}.png'.format(y))
        plt.close(fig)
        y = y + 1
# EcgToImages()

def ReadImages():
    """
    generator for loading images from folder 'data'
    """
    path2='data'
    images = glob.glob(path2 + '\*.png')
    for i in range(len(images)):
        yield cv2.imread(path2+'\{}.png'.format(i))
images = ReadImages() 


def CreateBoxes(signals, images):
    """
    creating annotations in the pascal voc format
    :param list signals: containing tuples with signal, and annotation 
    :param generator images: images in folder 'data'
    """
    a=0 #iterator
    for i in signals:
        img = next(images) #load image
        
        #localization of signal on image
        img2 = img[:, :, 0].reshape([img.shape[0], img.shape[1]])
        for j in range(img.shape[1]):
            if min(img2[:, img.shape[1] - 1 - j]) < 255:
                max_right = img.shape[1] - j
                break
        for j in range(img.shape[1]):
            if min(img2[:, j])< 255:
                max_left = j - 1
                break
            
        for j in range(img.shape[0]):
            if min(img2[j, : ]) < 255:
                max_top = j - 1
                break
        for j in range(img.shape[0]):
            if min(img2[img.shape[0]-1-j, : ])< 255:
                max_bot = img.shape[0]-j
                break
        
        max_x_pixels = max_right - max_left #lenght of signal in pixels
        max_y_pixels = max_bot - max_top  #hight of signal in pixels
        
        signal_max_x = i[0].shape[0] #number of signal samples
        signal_max_y = abs(max(i[0]) - min(i[0])) #hight of signal in mV
        
        scale_X = max_x_pixels / signal_max_x  
        scale_Y = max_y_pixels / signal_max_y 
        
        y = 0 #iterator
        features = i[1] 
        boolerP=False
        boolerN=False
        boolerT=False
        
        #read localization of waves
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
            
        #amplitude of the wave in pixels
        y_wave_botP = min(i[0][x_wave_beginP:x_wave_endP]) - min(i[0])
        y_wave_topP = max(i[0][x_wave_beginP:x_wave_endP]) - min(i[0])
    
        y_wave_botN = min(i[0][x_wave_beginN:x_wave_endN]) - min(i[0])
        y_wave_topN = max(i[0][x_wave_beginN:x_wave_endN]) - min(i[0])
        
        y_wave_botT = min(i[0][x_wave_beginT:x_wave_endT]) - min(i[0])
        y_wave_topT = max(i[0][x_wave_beginT:x_wave_endT]) - min(i[0])
        
        #determination of corner points
        pt1P = (max_left + int(x_wave_endP * scale_X), max_bot - int(y_wave_topP * scale_Y)-2)
        pt2P = (max_left + int(x_wave_beginP * scale_X), max_bot - int(y_wave_botP* scale_Y))
        
        pt1N = (max_left + int(x_wave_endN * scale_X), max_bot - int(y_wave_topN * scale_Y)-2)
        pt2N = (max_left + int(x_wave_beginN * scale_X), max_bot - int(y_wave_botN* scale_Y))
    
        pt1T = (max_left + int(x_wave_endT * scale_X), max_bot -  int(y_wave_topT * scale_Y)-2)
        pt2T = (max_left + int(x_wave_beginT * scale_X), max_bot -  int(y_wave_botT* scale_Y))
        
  
        path5='\data\\'
        #create XML in pascal VOC format 
        writer = Writer(path5+'{}.png'.format(a), 640, 480)
        
        writer.addObject('P', pt2P[0], pt1P[1], pt1P[0], pt2P[1])
        writer.addObject('QRS', pt2N[0], pt1N[1], pt1N[0], pt2N[1])
        writer.addObject('T', pt2T[0], pt1T[1], pt1T[0], pt2T[1])
       
        if a%6==0:
            writer.save('dat\\test\\annotations\{}.xml'.format(a))
            # file = md.parse("dat\\test\\annotations\{}.xml".format(a)) 
            # file.getElementsByTagName( "path" )[ 0 ].childNodes[ 0 ].nodeValue = '\content\drive\My Drive\dlaKuby\dat\\test\\images\{}.png'.format(a)
  
            # with open( "dat\\test\\annotations\{}.xml".format(a), "w" ) as fs:  
          
            #     fs.write( file.toxml() ) 
            #     fs.close()  
            # img=cv2.imread('D:\Jasiu\Dokumenty\studia\inzynierka\project\data\\{}.png'.format(a))
            # img_box = cv2.rectangle(img, (pt2P[0],pt1P[1]), (pt1P[0], pt2P[1]), (255, 30, 30), thickness=1)
            # img_box1 = cv2.rectangle(img,  (pt2N[0], pt1N[1]), (pt1N[0], pt2N[1]), (255, 30, 30), thickness=1)
            # img_box2 = cv2.rectangle(img, (pt2T[0], pt1T[1]), (pt1T[0], pt2T[1]), (255, 30, 30), thickness=1)
            # line1 = cv2.line(img, pt1=(max_left, max_bot), pt2=(max_left, max_top), color=(0,0,255), thickness=1)
            # line2 = cv2.line(img, pt1=(max_right, max_bot), pt2=(max_right, max_top), color=(0,0,255), thickness=1)
            # line3 = cv2.line(img, pt1=(max_left, max_top), pt2=(max_right, max_top), color=(0,0,255), thickness=1)
            # line4 = cv2.line(img, pt1=(max_left, max_bot), pt2=(max_right, max_bot), color=(0,0,255), thickness=1)
            
            # fig = plt.figure()
            # plt.imshow(img)
            # plt.imshow(img_box1)
            # plt.imshow(img_box2)
            # plt.imshow(line1)
            # plt.imshow(line2)
            # plt.imshow(line3)
            # plt.imshow(line4)
            
            
            # plt.axis('off')
            # plt.show()
            # plt.close()
            # a=a+1
            continue
        if a%3==0:
            writer.save('dat\\validation\\annotations\{}.xml'.format(a))
            # file = md.parse("dat\\validation\\annotations\{}.xml".format(a)) 
            # file.getElementsByTagName( "path" )[ 0 ].childNodes[ 0 ].nodeValue = '\content\drive\My Drive\dlaKuby\dat\\validation\\images\{}.png'.format(a)
  
            # with open( "dat\\validation\\annotations\{}.xml".format(a), "w" ) as fs:  
          
            #     fs.write( file.toxml() ) 
            #     fs.close()  
            # img=cv2.imread('D:\Jasiu\Dokumenty\studia\inzynierka\project\data\\{}.png'.format(a))
            # img_box = cv2.rectangle(img, (pt2P[0],pt1P[1]), (pt1P[0], pt2P[1]), (255, 30, 30), thickness=1)
            # img_box1 = cv2.rectangle(img,  (pt2N[0], pt1N[1]), (pt1N[0], pt2N[1]), (255, 30, 30), thickness=1)
            # img_box2 = cv2.rectangle(img, (pt2T[0], pt1T[1]), (pt1T[0], pt2T[1]), (255, 30, 30), thickness=1)
            # line1 = cv2.line(img, pt1=(max_left, max_bot), pt2=(max_left, max_top), color=(0,0,255), thickness=1)
            # line2 = cv2.line(img, pt1=(max_right, max_bot), pt2=(max_right, max_top), color=(0,0,255), thickness=1)
            # line3 = cv2.line(img, pt1=(max_left, max_top), pt2=(max_right, max_top), color=(0,0,255), thickness=1)
            # line4 = cv2.line(img, pt1=(max_left, max_bot), pt2=(max_right, max_bot), color=(0,0,255), thickness=1)
            
            # fig = plt.figure()
            # plt.imshow(img)
            # plt.imshow(img_box1)
            # plt.imshow(img_box2)
            # plt.imshow(line1)
            # plt.imshow(line2)
            # plt.imshow(line3)
            # plt.imshow(line4)
            
            # plt.axis('off')
            # plt.show()
            # plt.close()
        else:
            writer.save('dat\\train\\annotations\{}.xml'.format(a))
            # file = md.parse("dat\\train\\annotations\{}.xml".format(a)) 
            # file.getElementsByTagName( "path" )[ 0 ].childNodes[ 0 ].nodeValue = '\content\drive\My Drive\dlaKuby\dat\\train\\images\{}.png'.format(a)
  
            # with open("dat\\train\\annotations\{}.xml".format(a), "w" ) as fs:  
          
            #     fs.write( file.toxml() ) 
            #     fs.close()  
            # img=cv2.imread('D:\Jasiu\Dokumenty\studia\inzynierka\project\data\\{}.png'.format(a))
            # img_box = cv2.rectangle(img, (pt2P[0],pt1P[1]), (pt1P[0], pt2P[1]), (255, 30, 30), thickness=1)
            # img_box1 = cv2.rectangle(img,  (pt2N[0], pt1N[1]), (pt1N[0], pt2N[1]), (255, 30, 30), thickness=1)
            # img_box2 = cv2.rectangle(img, (pt2T[0], pt1T[1]), (pt1T[0], pt2T[1]), (255, 30, 30), thickness=1)
            # line1 = cv2.line(img, pt1=(max_left, max_bot), pt2=(max_left, max_top), color=(0,0,255), thickness=1)
            # line2 = cv2.line(img, pt1=(max_right, max_bot), pt2=(max_right, max_top), color=(0,0,255), thickness=1)
            # line3 = cv2.line(img, pt1=(max_left, max_top), pt2=(max_right, max_top), color=(0,0,255), thickness=1)
            # line4 = cv2.line(img, pt1=(max_left, max_bot), pt2=(max_right, max_bot), color=(0,0,255), thickness=1)
            
            # fig = plt.figure()
            # plt.imshow(img)
            # plt.imshow(img_box1)
            # plt.imshow(img_box2)
            # plt.imshow(line1)
            # plt.imshow(line2)
            # plt.imshow(line3)
            # plt.imshow(line4)
            
            # plt.axis('off')
            # plt.show()
            # plt.close()
        
        a=a+1
        
"""## Main"""


CreateBoxes(signals[:],images)

path="dat"
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=path)
trainer.setTrainConfig(object_names_array=["P","QRS","T"], batch_size=2, num_experiments=50,train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

# metrics = trainer.evaluateModel(model_path=path+"/models", json_path=path+"/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
# print(metrics)