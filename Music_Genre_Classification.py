# Importing necessary libraries

import pandas as pd
import numpy as np
import os
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
%matplotlib inline

# Providing filepath to access the music libraries

path = '/content/drive/MyDrive/Colab_Notebook/Music_Genre_Classification/Data/genres_original'
metadata = pd.read_csv('/content/drive/MyDrive/Colab_Notebook/Music_Genre_Classification/Data/features_30_sec.csv')
metadata.head()

# Here we convert signals to smaller frames. Then we try to discard the noise and the frequency kept is one with high probability of containing necessary information.


def extract(file):
    audio, sample_rate = librosa.load(req_file, res_type='kaiser_fast') 
    feat = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    feat_scaled = np.mean(feat.T,axis=0)
    
    return feat_scaled

# metadata.drop(labels=552, axis=0, inplace=True)

# Here we go through audio files and extract necessary features with the help of Mel-Frequency Cepstral Coefficients 

from tqdm import tqdm

extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    try:
        label = row["label"]
        req_file = os.path.join(os.path.abspath(path), label+'/',str(row["filename"]))    
        data = extract(req_file)
        extracted_features.append([data,label])
    except Exception as e:
        print(f"Error: {e}")
        continue

# Here the extracted features is converted to Pandad DF

feat_extracted_df = pd.DataFrame(extracted_features,columns=['feature','class'])
feat_extracted_df.head()

feat_extracted_df['class'].value_counts()

# Here we try to split the dataset into independent and dependent

X=np.array(feat_extracted_df['feature'].tolist())
y=np.array(feat_extracted_df['class'].tolist())

X.shape

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

y.shape

# Train - test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

X_train

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

# To find the num of classes

n = y.shape[1]

y.shape[1]

model=Sequential()
model.add(Dense(1024,input_shape=(40,), activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))

#To get final layer

model.add(Dense(n, activation="softmax"))


# Displays model summary

model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

import time
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)

# Training the model

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

epoc = 100
batch = 32

checkpointer = ModelCheckpoint(filepath=f'saved_models/audio_classification_{current_time}.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

history = model.fit(X_train, y_train, batch_size = batch, epochs = epoc, validation_data = (X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

model.evaluate(X_test,y_test,verbose=0)

pd.DataFrame(history.history).plot(figsize=(12,6))
plt.show()

model.predict_classes(X_test)

# Prediction

filename="/content/drive/MyDrive/Colab_Notebook/Music_Genre_Classification/audio/blues.00000.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
feat = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
feat_scaled = np.mean(feat.T,axis=0)

print(feat_scaled)
feat_scaled=feat_scaled.reshape(1,-1)
print(feat_scaled)
print(feat_scaled.shape)
predicted_label=model.predict_classes(feat_scaled)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label) 
prediction_class

