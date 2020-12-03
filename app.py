"""## Setup"""

import tkinter as tk 
from tkinter import filedialog
import wfdb
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
import sys
import numpy as np
from imageai.Detection.Custom import CustomObjectDetection
from keras.models import load_model
from PIL import Image
import glob
import cv2
import os
import pandas
from scipy import signal as sig

"""## Utils"""

def butter_filter(fd, fg, sampling_rate, order=4):

    nyq = 0.5 * sampling_rate
    low_freq = fd / nyq
    high_freq = fg / nyq
    if fd == 0:
        b_o, a_o = sig.butter(order, [high_freq], btype='low')
    elif fg == 0:
        b_o, a_o = sig.butter(order, [low_freq], btype='high')
    else:
        b_o, a_o = sig.butter(order, [low_freq, high_freq], btype='band')
    return b_o, a_o


def running_mean(input_signal, N):
    return np.convolve(input_signal, np.ones((N,))/N)[(N-1):]


def special_mean(input_signal, N, threshold=0):
    output_signal = np.zeros(len(input_signal))
    prev_mean = 0

    if threshold ==0:
        threshold = np.std(input_signal)

    for i, x in enumerate(input_signal, 1):
        if (i+N) >= len(input_signal):
            break;
        window = input_signal[i:i+N]
        local_min = np.min(window)
        local_max = np.max(window)
        local_mean = np.mean(window)

        if local_max-local_min < threshold:
            if prev_mean != 0:
                output_signal[i] = prev_mean
            else:
                output_signal[i] = local_mean
            prev_mean = local_mean
        else:
            output_signal[i] = input_signal[i]
            prev_mean = 0
    return output_signal


def Smooth(x, window_len=11, window='hanning'):
    """
    smoothing the signal - noise reduction
    :param array x: signal
    :param int window_len: 
    :param str window: type of window function
    :return: array y: smoothed signal
    """
    # x = x.T[0]
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

def SplitSignalToImages(signal):
    """    
    splitting the signals 
    :param list data: containing signal 
    """ 
    global fragments
    fragments=[]
    start=0 #start of fragment
    stop=500 #end of fragment
    print(int(signal.shape[0]/412))
    for i in range(int(signal.shape[0]/500)):
        fragments.append(signal[start:stop])
        start=start+500
        stop=stop+500
    ToImages(fragments)
    
def ToImages(fragments):
    """
    Transform signals 1d to images 2d and save them to appropriate folder
    """
    j=0
    for i in fragments:
        if j<len(fragments):      
            fig = plt.figure(figsize=(640/dpi,480/dpi), dpi=dpi)
            plt.ylim(min(signal), max(signal))
            plt.xlim(0, 500)
            plt.plot(i)
            plt.axis('off')
            fig.savefig('signal\{}.png'.format(j))
        j=j+1
    
    CombineImages(len(fragments),"img")
    
def RemoveFrame(img):
    """
    removing an unnecessary frame of images so that the signal is continuous
    """
    img2 = img[:, :, 0].reshape([img.shape[0], img.shape[1]])
    for j in range(img.shape[1]):
        if min(img2[:, img.shape[1] - 1 - j]) < 255:
            max_right = img.shape[1] - j
            print(max_right)
            break
    for j in range(img.shape[1]):
        if min(img2[:, j])< 255:
            max_left = j-1
            print(max_left)
            break
    img=img[:,79:490,:]
    
    return img

def CombineImages(numberOfFragments,typeOfImages):
    """
     combining signal so that the signal is continuous
    """
    images = ReadImages(typeOfImages)
    img = next(images)
    img = RemoveFrame(img)
    if numberOfFragments>1:
        for i in range(numberOfFragments):       
            if i>0:
                new_image = np.concatenate((img, img2), axis = 1)
                img=new_image
            try:
                img2 = next(images)
            except:
                print("error")
            img2 = RemoveFrame(img2)
        img = Image.fromarray(new_image, 'RGB')
        img.save('{}.png'.format(typeOfImages))
    else:
        img = Image.fromarray(img, 'RGB')
        img.save('{}.png'.format(typeOfImages))

def ReadImages(typeOfImages):
    """
    generator for loading images to combine
    """
    if(typeOfImages=="img"):
        images = glob.glob('signal\*.png')
        for i in range(len(images)):
            yield cv2.imread('signal\{}.png'.format(i))
    if(typeOfImages=="detimg"):
        images = glob.glob('signal\det*.png')
        for i in range(len(images)):
            yield cv2.imread('signal\det{}.png'.format(i))
     
def OpenSignal():
    """
    load signal 
    """
    files = glob.glob('signal/*')
    for f in files:
        os.remove(f)
    
    path=ReadFile()[:-4]
    global signal, fields
    extention = 'q1c'
    try:
            
        annotation = wfdb.rdann(path, extension=extention)
        sampfrom=min(annotation.sample)
        sampto=max(annotation.sample)
        signal, fields = wfdb.rdsamp(path, sampfrom=sampfrom, sampto=sampto, channels=[0])
        
    except:
        signal, fields = wfdb.rdsamp(path, channels=[0])
        
   
    signal=signal.flatten()

    b, a = butter_filter(1.0, 0, 50)
    filtered_data = sig.filtfilt(b, a, signal)  
    
    filtered_data2 = special_mean(filtered_data, 3)
    signal = special_mean(filtered_data2, 3)
    
    signal=signal[2000:2500]/100
    plt.plot(signal)
    plt.show()
    
    signal = sig.resample(signal, int(signal.shape[0]*250/fields['fs']))
    signal = Smooth(signal, window_len=10)
    plt.plot(signal)
    # signal=signal[0:2500]
    SplitSignalToImages(signal)
    img = tk.PhotoImage(file="img.png")
    global image_window
    image_window = ScrollableImage(window, image=img, scrollbarwidth=6, 
                               width=1400, height=400)
    image_window.place(relx = 0.5,  
                       rely = 0.4, 
                       anchor = 'center')
    
def Predict():
    """
    predict
    """
    Predict1()
    Predict2()
    
def Predict1(): 
    """
     predict location of waves
     """
    images = glob.glob('signal\*.png')
    if(len(images)!= 0):
        
        for i in range(len(images)):
            detector = CustomObjectDetection()
            detector.setModelTypeAsYOLOv3()
            detector.setModelPath("model12/dat/models/detection_model-ex-048--loss-0011.918.h5")
            detector.setJsonPath("model12/dat/json/detection_config.json")
            detector.loadModel()
            detections = detector.detectObjectsFromImage(input_image="signal\\{}.png".format(i), output_image_path="signal\\det{}.png".format(i))
            # for detection in detections:
            #     print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
        
        CombineImages(len(images),"detimg")
        img = tk.PhotoImage(file="detimg.png")
        image_window = ScrollableImage(window, image=img, scrollbarwidth=6, 
                               width=1400, height=400)
        image_window.place(relx = 0.5,  
                       rely = 0.4, 
                       anchor = 'center')
        
def Predict2(): # predictModel2
    """
    predict 5 parameters
    """
    if int(signal.shape[0]/600) ==0:
        signals = sig.resample(signal, 600)
    else:
        signals=signal
    fragments=np.empty(shape=(int(signals.shape[0]/600),600)) 
    for i in range(int(signals.shape[0]/600)):
        fragments[i]=signals[i*600:i*600+600]
    model = load_model('model2/model2-100epok.h5')

    predictions = model.predict(fragments.reshape([fragments.shape[0], 600, 1]))
    RR=0
    P=0
    QRS=0
    T=0
    amplitude=0 
    
    RR=sum(predictions[0])
    amplitude=sum(predictions[1])
    P=sum(predictions[2])
    QRS=sum(predictions[3])
    T=sum(predictions[4])
    
    RR=RR/(fragments.shape[0])
    P=P/fragments.shape[0]
    QRS=QRS/fragments.shape[0]
    T=T/fragments.shape[0]
    amplitude=amplitude/fragments.shape[0]
    
    label1.config(text = str(int(RR/250*60))+"/min")
    label3.config(text = str(int(P))+"ms")
    label5.config(text = str(int(QRS))+"ms")
    label7.config(text = str(int(T))+"ms")
    label9.config(text = str(format(float(amplitude), '.3f'))+"mV")

def ReadFile():
    """
    show an "Open" dialog box and return the path to the selected file
    :return: str filename: path to the selected file
    """
    filename = filedialog.askopenfilename() 
    return filename
    
def ConvertSignal():
    """
    Convert ECG signal from csv to WFDB
    """
    File = ReadFile()
    path=File[-4:]
    if path==".csv":
        try:
            fs = tk.simpledialog.askstring(title="Freq",
                                          prompt="Frequency:")
            name = tk.simpledialog.askstring(title="Name",
                                          prompt="Name of File:")
            data = pandas.read_csv(File, engine="python", delimiter=',')
            signal = data['ecg'].values
            signal= signal.reshape(signal.shape[0],1)    
            b = np.zeros((signal.shape[0], 1))
            signal = np.append(signal, b, axis=1)
            wfdb.wrsamp("convertedSignals/" + name,units=['mV', 'mV'], fs = float(fs), sig_name=['I', 'II'], 
                        p_signal=signal, fmt=['16', '16'])
        except:
            tk.messagebox.showinfo("Eror", message="Wrong in file")
    else:
        tk.messagebox.showinfo("Eror", message="Wrong file")

class ScrollableImage(tk.Frame):
    """
    Class to make scrollable image
    """
    def __init__(self, master=None, **kw):
        self.image = kw.pop('image', None)
        sw = kw.pop('scrollbarwidth', 10)
        super(ScrollableImage, self).__init__(master=master, **kw)
        self.cnvs = tk.Canvas(self, highlightthickness=0, **kw)
        self.cnvs.create_image(0, 0, anchor='nw', image=self.image)
        # Vertical and Horizontal scrollbars
        self.v_scroll = tk.Scrollbar(self, orient='vertical', width=sw)
        self.h_scroll = tk.Scrollbar(self, orient='horizontal', width=sw)
        # Grid and configure weight.
        self.cnvs.grid(row=0, column=0,  sticky='nsew')
        self.h_scroll.grid(row=1, column=0, sticky='ew')
        self.v_scroll.grid(row=0, column=1, sticky='ns')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # Set the scrollbars to the canvas
        self.cnvs.config(xscrollcommand=self.h_scroll.set, 
                           yscrollcommand=self.v_scroll.set)
        # Set canvas view to the scrollbars
        self.v_scroll.config(command=self.cnvs.yview)
        self.h_scroll.config(command=self.cnvs.xview)
        # Assign the region to be scrolled 
        self.cnvs.config(scrollregion=self.cnvs.bbox('all'))
        self.cnvs.bind_class(self.cnvs, "<MouseWheel>", self.mouse_scroll)

    def mouse_scroll(self, evt):
        if evt.state == 0 :
            self.cnvs.yview_scroll(-1*(evt.delta), 'units') # For MacOS
            self.cnvs.yview_scroll(int(-1*(evt.delta/120)), 'units') # For windows
        if evt.state == 1:
            self.cnvs.xview_scroll(-1*(evt.delta), 'units') # For MacOS
            self.cnvs.xview_scroll(int(-1*(evt.delta/120)), 'units') # For windows        
            
"""## Main"""
app = QApplication(sys.argv)
screen = app.screens()[0]
dpi = screen.physicalDotsPerInch()
app.quit()
       
window = tk.Tk() 
window.title('ECG Analyzer') 
window.geometry("1500x800")
window.resizable(False, False)

button1 = tk.Button(window, text='Convert ECG signal\nfrom csv to WFDB', width=25, height = 3, command=ConvertSignal, font=("Helvetica", 15)) 
button1.place(relx = 0.15,  
                   rely = 0.06, 
                   anchor = 'center') 

button2 = tk.Button(window, text='Load WFDB signal ECG', width=25, command=OpenSignal, font=("Helvetica", 15)) 
button2.place(relx = 0.5,  
                   rely = 0.03, 
                   anchor = 'center') 

button3 = tk.Button(window, text='Analyze signal ECG', width=25, command=Predict, font=("Helvetica", 15)) 
button3.place(relx = 0.5,  
                   rely = 0.1, 
                   anchor = 'center') 

label = tk.Label( window, text = "Pulse: " , font=("Helvetica", 25))
label.place(relx = 0.42,  
                   rely = 0.68, 
                   anchor = 'center') 

label1 = tk.Label( window, text = "0/min " , font=("Helvetica", 25))
label1.place(relx = 0.7,  
                   rely = 0.68, 
                   anchor = 'center') 

label2 = tk.Label( window, text = "Average length of P wave: " , font=("Helvetica", 25))
label2.place(relx = 0.4,  
                   rely = 0.741, 
                   anchor = 'center') 
label3 = tk.Label( window, text = "0 ms " , font=("Helvetica", 25))
label3.place(relx = 0.7,  
                   rely = 0.741, 
                   anchor = 'center') 

label4 = tk.Label( window, text = "Average length of QRS complex: " , font=("Helvetica", 25))
label4.place(relx = 0.4,  
                   rely = 0.802, 
                   anchor = 'center') 
label5 = tk.Label( window, text = "0 ms " , font=("Helvetica", 25))
label5.place(relx = 0.7,  
                   rely = 0.802, 
                   anchor = 'center') 

label6 = tk.Label( window, text = "Average length of T wave: " , font=("Helvetica", 25))
label6.place(relx = 0.4,  
                   rely = 0.864, 
                   anchor = 'center') 
label7 = tk.Label( window, text = "0 ms " , font=("Helvetica", 25))
label7.place(relx = 0.7,  
                   rely = 0.864, 
                   anchor = 'center') 

label8 = tk.Label( window, text = "Average amplitude of QRS complex: " , font=("Helvetica", 25))
label8.place(relx = 0.4,  
                   rely = 0.925, 
                   anchor = 'center') 
label9 = tk.Label( window, text = "0 mV " , font=("Helvetica", 25))
label9.place(relx = 0.7,  
                   rely = 0.925, 
                   anchor = 'center') 
window.mainloop() 