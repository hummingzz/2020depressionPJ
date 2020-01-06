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

def model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    y_train_pred = np.argmax(model.predict(X_train), axis=1)

    y_test_pred_proba = model.predict(X_test)
    y_train_pred_proba = model.predict(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:, 1]

    # Computing confusion matrix for test dataset
    # conf_matrix = standard_confusion_matrix(y_test_1d, y_test_pred)
    conf_matrix = confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, conf_matrix

def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

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

data = pd.read_excel('/home/pehuang/zhaozhang/beijing/total-Final.xlsx')

data = data[['ID','TYPE']]

data.loc[data['TYPE'] == '抑郁', 'TYPE'] = 1
data.loc[data['TYPE'] == '健康', 'TYPE'] = 0
data.loc[data['TYPE'] == '正常', 'TYPE'] = 0

id_healthy = data[data['TYPE'] == 0].reset_index()
id_depressed = data[data['TYPE'] == 1].reset_index()

"""  随机采样健康人的id 抑郁人的id    """

random.seed(22)
np.random.seed(22)

slice1 = random.sample(range(len(id_depressed)), 25)
slice2 = random.sample(range(len(id_healthy)), 25)

slice3 = []
slice4 = []

for item in range(len(id_depressed)):
    if item not in slice1:
        slice3.append(item)
for item in range(len(id_healthy)):
    if item not in slice2:
        slice4.append(item)


test_index = pd.concat((id_depressed.ix[slice1],id_healthy.ix[slice2]),axis=0,ignore_index=True)
train_index = pd.concat((id_depressed.ix[slice3],id_healthy.ix[slice4]),axis=0,ignore_index=True)

print(train_index)
print(test_index)

train_id = train_index[['ID','TYPE']].rename(columns={"ID" : 'partic_id'})
test_id = test_index[['ID','TYPE']].rename(columns={"ID" : 'partic_id'})

print(train_id.shape)
print(test_id.shape)



""" deep learning train test dataset split  for deep learning """
"""   从spec 二维矩阵中 进行 深度学习 训练好模型 存放到固定位置 """

"""  存放输入数据的位置 """
spec_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/specspace'
""" 存放 partic id 的位置 仅存放了partic ID  """
spec_label_path = '/home/pehuang/zhaozhang/beijing/data07_tf/speclabel'
"""  存在 TYPE 以及部分其他feature  """
label_csv_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/csvspace'

label_train = train_id['partic_id']
label_train = set(list(label_train))
label_test = test_id['partic_id']
label_test = set(list(label_test))

data_train = np.arange(39*513).reshape(1,39,513)
data_test = np.arange(39*513).reshape(1,39,513)
data_train_label = np.arange(1).reshape(-1,)
data_test_label = np.arange(1).reshape(-1,)

"""   先选定我们要挑的id 因为id 对应的索引位置和spec、label都是对应且一致的   """
for file in os.listdir(spec_label_path):
    judgement = file[-10:-4]
    file_path = os.path.join(spec_label_path,file)
    data = list(np.load(file_path))
    selected_train_id = []
    selected_test_id = []
    for i in range(len(data)):
        if data[i] in label_train:
            selected_train_id.append(i)
        if data[i] in label_test:
            selected_test_id.append(i)

    print(len(selected_train_id))
    print(len(selected_test_id))



    # """ 挑选对应的spec 放入对应的训练集合 测试集合 """
    for file in os.listdir(spec_path):
        if file[-10:-4] == judgement:
            file_path = os.path.join(spec_path,file)
            data = np.load(file_path)
            print(data.shape)
            data_fortrain = data[selected_train_id,:]
            data_fortest = data[selected_test_id,:]
            print(data_fortrain.shape)
            print(data_fortest.shape)

    data_train = np.concatenate((data_train,data_fortrain), axis=0)
    print(data_train.shape)
    data_test = np.concatenate((data_test,data_fortest),axis=0)
    print(data_test.shape)


    """   挑选对应的label   用于模型train """

    for file in os.listdir(label_csv_path):
        if file[-10:-4] == judgement:
            file_path = os.path.join(label_csv_path, file)
            data = pd.read_csv(file_path)['TYPE'].values
            data_fortrain = data[selected_train_id]
            data_fortest = data[selected_test_id]
            print(data_fortrain.shape)
            print(data_fortest.shape)

    data_train_label = np.concatenate((data_train_label,data_fortrain), axis=0)
    print(data_train_label.shape)
    data_test_label = np.concatenate((data_test_label,data_fortest),axis=0)
    print(data_test_label.shape)


data_train = data_train[1:,:]
data_test = data_test[1:,:]

data_train_label = data_train_label[1:]
data_test_label = data_test_label[1:]

print(data_train.shape)
print(data_test.shape)
print(data_train_label.shape)
print(data_test_label.shape)


nb_classes = 2
y_train = np_utils.to_categorical(data_train_label, nb_classes)
y_test = np_utils.to_categorical(data_test_label, nb_classes)

X_train, X_test = data_train, data_test



# """  对原始数据取对数  """
# import math
# #
# # def f2(x):
# #     return math.log(x)
# # f1 = np.vectorize(f2)
# #
# # X_train = X_train+0.0000001
# # X_train = f1(X_train)
# # X_test = X_test+0.0000001
# # X_test = f1(X_test)
# # print(X_train.shape)
# # print(X_test.shape)

def scheduler(epoch):
    if epoch % 60 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print('lr changed to {}'.format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

model = Sequential()
model.add(Conv1D(128, 4,strides=1, activation='relu',padding="same", input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Attention())
model.add(Dense(2,activation='softmax'))

adam = optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

batch_size = 2048
epochs = 200

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, y_test), class_weight='auto', shuffle=True, callbacks=[reduce_lr])

score_train = model.evaluate(X_train, y_train, verbose=0)
print('Train accuracy:', score_train[1])
score_test = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', score_test[1])

print('Evaluating model...')
y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, conf_matrix = model_performance(model, X_train, X_test, y_train, y_test)

""" plot ROC AUC """
from sklearn.metrics import roc_curve, auc
y_pred_keras = model.predict(X_test)
print(y_test.shape)
print(y_pred_keras.shape)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test.argmax(axis=1), y_pred_keras.argmax(axis=1))
auc_keras = auc(fpr_keras, tpr_keras)

# plt.figure()
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.savefig('/home/pehuang/zhaozhang/beijing/visualization/'+'log_ROC_1.png')

'''custom evaluation metrics'''

print('Calculating additional test metrics...')
accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1-Score: {}".format(f1_score))

""" save model&plot model and accuracy&loss change figure"""

model.save('/home/pehuang/zhaozhang/beijing/visualization/model_1216.h5')
# plot_model(model,to_file='/home/pehuang/zhaozhang/beijing/spec1/model.png')

# acc = np.array(history.history['acc'])
# val_acc = np.array(history.history['val_acc'])
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# np.save('/home/pehuang/zhaozhang/beijing/visualization/acc1.npy',acc)
# np.save('/home/pehuang/zhaozhang/beijing/visualization/val_acc1.npy',val_acc)
# np.save('/home/pehuang/zhaozhang/beijing/visualization/loss1.npy',loss)
# np.save('/home/pehuang/zhaozhang/beijing/visualization/val_loss1.npy',val_loss)

# fig = plt.figure()
# plt.plot(history.history['acc'],label='training acc')
# plt.plot(history.history['val_acc'],label='val acc')
# plt.title('Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# fig.savefig('/home/pehuang/zhaozhang/beijing/visualization/'+'loglstm'+'accuracy_1.png')
#
# fig = plt.figure()
# plt.plot(history.history['loss'],label='training loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.title('Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# fig.savefig('/home/pehuang/zhaozhang/beijing/visualization/'+'loglstm'+'loss_1.png')
