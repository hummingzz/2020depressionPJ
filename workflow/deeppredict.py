from keras.models import Sequential
from keras import applications
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import os
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model,Model
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

np.random.seed(22)

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

class AttentionLayer(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(input_shape[-1], 1),
                                 initializer=self.init,
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x):
        # refer to the original paper
        # link: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

        # RNN 구조를 거쳐서 나온 hidden states (x)에 single layer perceptron (tanh activation)
        # 적용하여 나온 벡터가 uit
        u_it = K.tanh(K.dot(x, self.W) + self.b)

        # uit와 uw (혹은 us) 간의 similarity를 attention으로 사용
        # softmax를 통해 attention 값을 확률 분포로 만듬
        a_it = K.dot(u_it, self.u)
        a_it = K.squeeze(a_it, -1)
        a_it = K.softmax(a_it)

        return a_it

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

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


model = load_model('/home/pehuang/zhaozhang/beijing/visualization/model_1.h5',custom_objects={"Attention":Attention,"AttentionLayer":AttentionLayer, 'AttentivePoolingLayer':AttentivePoolingLayer})
# model = load_model('/home/pehuang/zhaozhang/beijing/spec1/model_1.h5')
print(model.summary())

input()

# model_feature = Model(inputs=model.input, outputs=model.get_layer('bidirectional_1').output)
model_feature = Model(inputs=model.input, outputs=model.get_layer('attention_1').output)

print(model_feature.summary())

npy_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/specspace'
csv_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/csvspace'
for file in os.listdir(npy_path):
    input_path = os.path.join(npy_path,file)
    feature = np.load(input_path)
    feature = pd.DataFrame(model_feature.predict(feature))
    print(file[7:-4])
    for csvfile in os.listdir(csv_path):
        if csvfile[5:-4] == file[7:-4]:
            input_path = os.path.join(csv_path, csvfile)
            label = pd.read_csv(input_path)
            full = pd.concat((label,feature), axis=1)
            full = full.groupby(['partic_id']).mean()
            full.to_csv('/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/fullspace_1/' + csvfile[5:-4] + '.csv')