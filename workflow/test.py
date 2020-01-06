import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import Lasso,LassoCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from imblearn import over_sampling
from collections import Counter
import random
import os
from keras.models import Sequential
from keras import applications
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils import np_utils,plot_model
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import sys
import glob
import argparse
import matplotlib.pyplot as plt

class AttentionLayer(Layer):
    '''
    Attention layer.
    '''

    def __init__(self, W_regularizer=None, b_regularizer=None, **kwargs):
        self.supports_masking = False
        # self.init = initializations.get(init)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        super(AttentionLayer, self).__init__(3**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Uw = self.add_weight(name='Uw',
                                  shape=((input_dim, 1)),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.trainable_weights = [self.Uw]
        super(1, self).build(input_shape)

    def compute_mask(self, input, mask):
        return mask

    def call(self, x, mask=None):
        print(K.int_shape(x))  # (None, 80, 200)
        print(K.int_shape(self.Uw))  # (200, 1)
        multData = K.exp(K.dot(x, self.Uw))
        if mask is not None:
            multData = mask * multData
        output = multData / (K.sum(multData, axis=1) + K.epsilon())[:, None]
        print(K.int_shape(output))  # (None, 80, 1)
        return output

    def get_output_shape_for(self, input_shape):
        newShape = list(input_shape)
        newShape[-1] = 1
        return tuple(newShape)

class AttentivePoolingLayer(Layer):

    from keras.initializers import he_uniform
    def __init__(self,W_regularizer=None,b_regularizer=None,**kwargs):
        self.supports_masking =False
        # self.mask =mask
        self.W_regularizer =regularizers.get(W_regularizer)
        self.b_regularizer =regularizers.get(b_regularizer)
        super(AttentivePoolingLayer, self).__init__(**kwargs)
    def build(self, input_shape):

        n_in =input_shape[2]
        n_out =1
        lim =np.sqrt(6./(n_in+n_out))
        # tanh initializer xavier
        self.W =K.random_uniform_variable((n_in,n_out),-lim,lim,
                                         name='{}_W'.format(self.name) )
        self.b =K.zeros((n_out,),name='{}_b'.format(self.name))
        self.trainable_weights=[self.W,self.b]
        self.regularizer =[]
        if self.W_regularizer is not None:
            self.add_loss(self.W_regularizer(self.W))
        if self.b_regularizer is not None:
            self.add_loss(self.b_regularizer(self.b))
        self.build =True
    def call(self, inputs,mask=None):

        memory =inputs
        print('memory shape',K.int_shape(memory))
        gi =K.tanh(K.dot(memory,self.W)+self.b)  #32 *6 *1
        gi =K.sum(gi,axis=-1)   # 32 *6
        alfa =K.softmax(gi)
        self.alfa =alfa
        output =K.sum(memory*K.expand_dims(alfa,axis=-1),axis=1) #sum(32 *6 *310)
        print('output..shape',K.int_shape(output))
        return output
    def compute_output_shape(self, input_shape):
        shape =input_shape
        shape =list(shape)

        return  (shape[0],shape[2])

    def compute_mask(self, inputs, mask=None):
        return None

class Attention(Layer):
	def __init__(self, regularizer=None, **kwargs):
		super(Attention, self).__init__(**kwargs)
		self.regularizer = regularizer
		self.supports_masking = True

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.context = self.add_weight(name='context',
									   shape=(input_shape[-1], 1),
									   initializer=initializers.RandomNormal(
									   		mean=0.0, stddev=0.05, seed=None),
									   regularizer=self.regularizer,
									   trainable=True)
		super(Attention, self).build(input_shape)

	def call(self, x, mask=None):
		attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
		attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)

		if mask is not None:
			# use only the inputs specified by the mask
			# import pdb; pdb.set_trace()
			attention = attention*K.cast(mask, 'float32')

		weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
		return weighted_sum

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

model = Sequential()
model.add(Conv1D(256, 1,strides=4, activation='relu',padding="same", input_shape=(39,513)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Attention())
model.add(Dense(128))
# model.add(AttentivePoolingLayer())
model.add(Dense(2,activation='softmax'))

adam = optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())









