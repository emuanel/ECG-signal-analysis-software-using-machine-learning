"""## Setup"""
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from keras.layers import Dense, Input, Activation, Conv1D, Flatten, MaxPooling1D, BatchNormalization, Convolution1D, Dropout, SeparableConv1D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

"""## Utils"""
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

def Fragmentation(data):
    """
    splitting signals into fragments of 600 samples adding 200 samples 
    :param list data: containing tuples with signal and annotation
    :return: list data: containing more tuples with signal, annotation
    """
    signals = []

    for i in data:
        stop = 600 #sample of signal to stop
        start = 0 #sample of signal to start
        start1 = 0 #index of annotation to start
        stop1 = 0 #index of annotation to stop
        booler = True 
        signal = i[0]
        features = i[1].reset_index()
        while stop < len(signal):
            unlabelled = True 
            #splitting annonations
            for inde, index, symbol in zip(features.index, features['index'], features['symbol']):
                if index >= start and index < stop and booler == True:
                    start1 = inde
                    booler = False
                    unlabelled = False
                if index > start and index < stop:
                    stop1 = inde
                    unlabelled = False
            annotation = features[start1:stop1 + 1]
            annotation = annotation.copy()
            annotation['index'] = annotation['index'] - start #adaptation to the new signal
            if (unlabelled == False):
                signals.append((signal[start:stop], annotation))
            
            #shift 200 samples
            start = start + 200
            stop = stop + 200
            booler = True
    return signals

def Labels(signals):
    """
    create new adnotations: 'avarage of interval R-R', 'avarage of QRS amplitude', 
    'avarage of P length', 'avarage of T length', 'avarage of QRS length'
    :param list signals: containing tuples with signal and annotation
    """
    data = []
    for i in signals:
        booler1 = False
        booler2 = False
        booler3 = False
        features = i[1]
        signal = i[0]
        Rpeaks = [] #lenghts of R wave in signal
        QRSamplitudes = [] #amplitudes of QRS comlexes in signal
        Plength = [] #lenghts of P waves in signal
        Tlength = [] #lenghts of R waves in signal
        QRSlength = [] #lenghts of R waves in signal
        previousIndex = 0
        previousIndex2 = 0
        firstR = True
        
        #new annotations
        labels = pd.DataFrame(columns=['heart rate', 'QRS amplitude', 'P length', 'T length', 'QRS length']) 

        for inde, index, symbol in zip(features.index, features['index'], features['symbol']):
            if (symbol == 'N'):
                QRSamplitudes.append(signal[index])
                if (firstR == False):
                    Rpeaks.append(index - previousIndex2)
                    previousIndex2 = index
                if (firstR == True):
                    previousIndex2 = index
                    firstR = False
            
            if (booler1):
                stop = index
                Plength.append(stop - start)
                booler1 = False
            if (symbol == 'p'):
                start = previousIndex
                booler1 = True

            if (booler2):
                stop = index
                Tlength.append(stop - start)
                booler2 = False
            if (symbol == 't'):
                start = previousIndex
                booler2 = True

            if (booler3):
                stop = index
                QRSlength.append(stop - start)
                booler3 = False
            if (symbol == 'N'):
                start = previousIndex
                booler3 = True

            previousIndex = index
        #counting avarage values of new labels
        if (len(Rpeaks) != 0):
            labels.at[0, 'heart rate'] = float(sum(Rpeaks) / len(Rpeaks))
        if (len(QRSamplitudes) != 0):
            labels.at[0, 'QRS amplitude'] = float(sum(QRSamplitudes) / len(QRSamplitudes))
        if (len(Plength) != 0):
            labels.at[0, 'P length'] = sum(Plength) / len(Plength)
        if (len(Tlength) != 0):
            labels.at[0, 'T length'] = sum(Tlength) / len(Tlength)
        if (len(QRSlength) != 0):
            labels.at[0, 'QRS length'] = sum(QRSlength) / len(QRSlength)
        data.append((i[0], labels))
    return data


def RemoveSignals():
    """    
    removing signals that do not have information about any of the parameters
    """
    for i in range(len(data) - 1, -1, -1):
        if np.isnan(data[i][1].to_numpy(float)).any() or data[i][1].empty:
            data.pop(i)




# def SignalLengthEqualization():
#     iterator = 0
#     for i in data_X:
#         data_X[iterator] = i[:900]
#         iterator = iterator + 1


def BuildModel():
    """    
    building cnn1d model with 5 outputs
    """
    inputSignal = Input(shape=(600, 1), name='ImageInput')
    x = Conv1D(32, 3, activation='relu', padding='same', name='Conv1_1')(inputSignal)
    x = Conv1D(32, 3, activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling1D(2, name='pool1')(x)
    x = SeparableConv1D(32, 3, activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv1D(32, 3, activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling1D(2, name='pool2')(x)
    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv1D(64, 3, activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    x1 = SeparableConv1D(128, 3, activation='relu', padding='same')(x)
    x1 = SeparableConv1D(128, 3, activation='relu', padding='same')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = SeparableConv1D(256, 3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = SeparableConv1D(256, 3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = SeparableConv1D(256, 3, activation='relu', padding='same')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Dropout(0.6)(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(1)(x1)
    
    x2 = SeparableConv1D(64, 3, activation='relu', padding='same')(x)
    x2 = SeparableConv1D(64, 3, activation='relu', padding='same')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = SeparableConv1D(128, 3, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = SeparableConv1D(128, 3, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = SeparableConv1D(128, 3, activation='relu', padding='same')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(0.6)(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(1)(x2)
    
    x3 = SeparableConv1D(64, 3, activation='relu', padding='same')(x)
    x3 = SeparableConv1D(64, 3, activation='relu', padding='same')(x3)
    x3 = MaxPooling1D(2)(x3)
    x3 = SeparableConv1D(128, 3, activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = SeparableConv1D(128, 3, activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = SeparableConv1D(128, 3, activation='relu', padding='same')(x3)
    x3 = MaxPooling1D(2)(x3)
    x3 = Flatten()(x3)
    x3 = Dense(256, activation='relu')(x3)
    x3 = Dropout(0.6)(x3)
    x3 = Dense(256, activation='relu')(x3)
    x3 = Dropout(0.5)(x3)
    x3 = Dense(1)(x3)

    x4 = SeparableConv1D(64, 3, activation='relu', padding='same')(x)
    x4 = SeparableConv1D(64, 3, activation='relu', padding='same')(x4)
    x4 = MaxPooling1D(2)(x4)
    x4 = SeparableConv1D(128, 3, activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = SeparableConv1D(128, 3, activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = SeparableConv1D(128, 3, activation='relu', padding='same')(x4)
    x4 = MaxPooling1D(2)(x4)
    x4 = Flatten()(x4)
    x4 = Dense(256, activation='relu')(x4)
    x4 = Dropout(0.6)(x4)
    x4 = Dense(256, activation='relu')(x4)
    x4 = Dropout(0.5)(x4)
    x4 = Dense(1)(x4)
    
    x5 = SeparableConv1D(64, 3, activation='relu', padding='same')(x)
    x5 = SeparableConv1D(64, 3, activation='relu', padding='same')(x5)
    x5 = MaxPooling1D(2)(x5)
    x5 = SeparableConv1D(128, 3, activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = SeparableConv1D(128, 3, activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = SeparableConv1D(128, 3, activation='relu', padding='same')(x5)
    x5 = MaxPooling1D(2)(x5)
    x5 = Flatten()(x5)
    x5 = Dense(256, activation='relu')(x5)
    x5 = Dropout(0.6)(x5)
    x5 = Dense(256, activation='relu')(x5)
    x5 = Dropout(0.5)(x5)
    x5 = Dense(1)(x5)

    model = Model(inputs=inputSignal, outputs=(x1,x2,x3,x4,x5))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model

"""## Main"""
path1='qt-database-1.0.0\\'
records=glob.glob(path1+"*.dat")
data=Segmentation(records[:]) 
    
for i in data:
    i = (Smooth(i[0]), i[1])
    
data2 = Fragmentation(data[:])

data = Labels(data2)

RemoveSignals()

data_X = []
data_Y = []
for i in data:
    data_X.append(i[0])
    data_Y.append(i[1])

data_X = np.array(data_X).reshape([6312, 600, 1])
data_Y = pd.concat(data_Y)

X_train, X_valid, y_train, y_valid = train_test_split(data_X, data_Y, test_size=0.3, random_state=1)
X_test, X_valid, y_test, y_valid = train_test_split(X_valid, y_valid, test_size=0.5, random_state=1)
y_train = [y_train['heart rate'], y_train['QRS amplitude'], y_train['P length'], y_train['T length'], y_train['QRS length']]
y_valid = [y_valid['heart rate'], y_valid['QRS amplitude'], y_valid['P length'], y_valid['T length'], y_valid['QRS length']]
y_test = [y_test['heart rate'], y_test['QRS amplitude'], y_test['P length'], y_test['T length'], y_test['QRS length']]

model = BuildModel()
epochs = 100

# history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), batch_size=32)
model.save("model2-100epok.h5")

plt.subplot(2, 3, 1)
plt.plot(list(range(epochs)), history.history['dense_3_loss'])
plt.plot(list(range(epochs)), history.history['val_dense_3_loss'])
plt.title("Heart rate")

plt.subplot(2, 3, 2)
plt.plot(list(range(epochs)), history.history['dense_6_loss'])
plt.plot(list(range(epochs)), history.history['val_dense_6_loss'])
plt.title("QRS amplitude")

plt.subplot(2, 3, 3)
plt.plot(list(range(epochs)), history.history['dense_9_loss'])
plt.plot(list(range(epochs)), history.history['val_dense_9_loss'])
plt.title("P length")

plt.subplot(2, 3, 4)
plt.plot(list(range(epochs)), history.history['dense_12_loss'])
plt.plot(list(range(epochs)), history.history['val_dense_12_loss'])
plt.title("QRS length")

plt.subplot(2, 3, 5)
plt.plot(list(range(epochs)), history.history['dense_15_loss'])
plt.plot(list(range(epochs)), history.history['val_dense_15_loss'])
plt.title("T length")
plt.show()

from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn
model = load_model('model2-100epok.h5')
y_true = y_test[0]
y_pred = model.predict(X_test)
print("MSE R-R : ",mean_squared_error(y_true, y_pred[0]))
print("RMSE R-R : ",mean_squared_error(y_true, y_pred[0], squared = False))
print("MAE R-R : ",mean_absolute_error(y_true, y_pred[0]))
print("R2 score R-R : ",r2_score(y_true, y_pred[0]))
print()

y_true = y_test[1]
print("MSE QRS amplitude: ", mean_squared_error(y_true, y_pred[1]))
print("RMSE QRS amplitude: ",mean_squared_error(y_true, y_pred[1], squared = False))
print("MAE QRS amplitude: ",mean_absolute_error(y_true, y_pred[1]))
print("R2 QRS amplitude: ",r2_score(y_true, y_pred[1]))
print()

y_true = y_test[2]
print("MSE P lenght: ", mean_squared_error(y_true, y_pred[2]))
print("RMSE P lenght: ",mean_squared_error(y_true, y_pred[2], squared = False))
print("MAE P lenght: ",mean_absolute_error(y_true, y_pred[2]))
print("R2 score P lenght: ",r2_score(y_true, y_pred[2]))
print()

y_true = y_test[3]
print("MSE T lenght: ", mean_squared_error(y_true, y_pred[3]))
print("RMSE T lenght: ",mean_squared_error(y_true, y_pred[3], squared = False))
print("MAE T lenght: ",mean_absolute_error(y_true, y_pred[3]))
print("R2 score T lenght: ",r2_score(y_true, y_pred[3]))
print()

y_true = y_test[4]
print("MSE QRS lenght: ", mean_squared_error(y_true, y_pred[4]))
print("RMSE QRS lenght: ",mean_squared_error(y_true, y_pred[4], squared = False))
print("MAE QRS lenght: ",mean_absolute_error(y_true, y_pred[4]))
print("R2 score QRS lenght: ",r2_score(y_true, y_pred[4]))











