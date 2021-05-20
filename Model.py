import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Input, LSTM, Reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras import backend as K

#To use F1 score as metric
#Based on https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def CNN():
    embed_size=100
    model_input=Input(shape=(None,1))
    #model=Sequential()
    #model.add(Embedding(vocab_size, 32))
    #model.add(Dropout(0.2))
    submodels=[]
    for kw in (3,4,5):
        conv=Conv1D(100, kernel_size=(kw*embed_size,), padding='valid', activation='relu', kernel_regularizer=l2(3),strides=embed_size)(model_input)
        conv=GlobalMaxPooling1D()(conv)
        #conv=Flatten()(conv)
        submodels.append(conv)
    z=Concatenate()(submodels)
    z=Dropout(0.5)(z)
    model_output=Dense(1,activation='sigmoid')(z)
    model=Model(model_input, model_output)
    model.summary()
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc',f1_m,precision_m, recall_m])
    return model

def CRNN():
    #model_input=Input(batch_input_shape=(1,None,1))
    model_input=Input(shape=(None,1))
    submodels=[]
    embed_size=100
    for kw in (3,4,5):
        conv=Conv1D(100, kernel_size=(kw*embed_size,), padding='valid', activation='relu', kernel_regularizer=l2(3),strides=embed_size)(model_input)
        conv=GlobalMaxPooling1D()(conv)
        #conv=Flatten()(conv)
        submodels.append(conv)
    z=Concatenate()(submodels)
    z=Dropout(0.5)(z) # (None, 300)
    z=Reshape(target_shape=((1,300)))(z)
    z=LSTM(128, return_sequences=False,stateful=False)(z)

    model_output=Dense(1,activation='sigmoid')(z)

    model=Model(model_input, model_output)
    model.summary()
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc',f1_m,precision_m, recall_m])
    return model
