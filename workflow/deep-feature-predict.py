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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import sys
import glob
import argparse
import matplotlib.pyplot as plt


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

"""     prepare acoustic features and deep learned features   """

def coral(Xs, Xt):
    '''
    Perform CORAL on the source domain features
    :param Xs: ns * n_feature, source feature
    :param Xt: nt * n_feature, target feature
    :return: New source domain features
    '''
    cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
    cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
    A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                     scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
    Xs_new = np.dot(Xs, A_coral).astype(float)

    return Xs_new

# audio_data_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/audiofeaturespace'
deep_data_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/fullspace_1'

data_train = np.arange(1*132).reshape(1,132)
data_test = np.arange(1*132).reshape(1,132)
# data_train = np.arange(1*14).reshape(1,14)
# data_test = np.arange(1*14).reshape(1,14)

for file in os.listdir(deep_data_path):
    file_path = os.path.join(deep_data_path, file)
    deep_feature = pd.read_csv(file_path)
    # deep_feature.drop(columns=['TYPE', 'BDI', 'AGE'], axis=1, inplace=True)
    # feature = pd.merge(audio_feature, deep_feature, on='partic_id')
    # feature = pd.concat((audio_feature,deep_feature),axis=1)
    feature = deep_feature
    print(feature.head())
    print(feature.shape)
    feature_for_train = pd.merge(feature, train_id['partic_id'], how='inner', on='partic_id')
    feature_for_test = pd.merge(feature, test_id['partic_id'], how='inner', on='partic_id')
    print(feature_for_test.head())
    data_train = np.concatenate((data_train, feature_for_train.values), axis=0)
    data_test = np.concatenate((data_test, feature_for_test.values), axis=0)

# for file in os.listdir(audio_data_path):
#     if file not in ['data_full_model_07_new.csv']:
#         judgement = file[-6:-4]
#         file_path = os.path.join(audio_data_path,file)
#         audio_feature = pd.read_csv(file_path)
#         for file in os.listdir(deep_data_path):
#             if file[:2]  == judgement:
#                 file_path = os.path.join(deep_data_path,file)
#                 deep_feature = pd.read_csv(file_path)
#                 deep_feature.drop(columns=['TYPE','BDI','AGE'],axis=1,inplace=True)
#                 feature = pd.merge(audio_feature,deep_feature,on='partic_id')
#                 # feature = pd.concat((audio_feature,deep_feature),axis=1)
#                 print(feature.head())
#                 print(feature.shape)
#                 feature_for_train = pd.merge(feature, train_id['partic_id'], how='inner', on='partic_id')
#                 feature_for_test = pd.merge(feature, test_id['partic_id'], how='inner', on='partic_id')
#                 print(feature_for_test.head())
#                 data_train = np.concatenate((data_train,feature_for_train.values), axis=0)
#                 data_test = np.concatenate((data_test, feature_for_test.values), axis=0)

data_train = data_train[1:,:]
data_test = data_test[1:,:]
data_train = pd.DataFrame(data_train, columns=feature_for_train.columns)
data_test = pd.DataFrame(data_test, columns=feature_for_test.columns)

print(data_train.head())
print(data_test.head())
print(data_train.shape)
print(data_test.shape)

train_data = data_train
test_data = data_test
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

""" preprocess """
train_label = train_data[['TYPE', 'BDI']]
test_label = test_data[['TYPE', 'BDI']]

train_data.drop(columns=['partic_id', 'TYPE', 'BDI'], axis=1, inplace=True)
test_data.drop(columns=['partic_id',  'TYPE', 'BDI'], axis=1, inplace=True)

# train_data = pd.get_dummies(train_data, columns=['SEX'])
# test_data = pd.get_dummies(test_data, columns=['SEX'])

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

sc = StandardScaler()
mami = MinMaxScaler()
rb = RobustScaler()

feature_src = np.array(train_data, dtype=np.float)
feature_tar = np.array(test_data, dtype=np.float)

where_are_nan = np.isnan(feature_src)
where_are_inf = np.isinf(feature_src)
feature_src[where_are_nan] = 0
feature_src[where_are_inf] = 0
feature_src = mami.fit_transform(feature_src)

where_are_nan = np.isnan(feature_tar)
where_are_inf = np.isinf(feature_tar)
feature_tar[where_are_nan] = 0
feature_tar[where_are_inf] = 0
feature_tar = mami.fit_transform(feature_tar)

train_label = train_label.values
src_label = train_label[:, 0]
test_label = test_label.values
tar_label = test_label[:, 0]

print('Before resampled dataset shape %s' % Counter(src_label))
sm = over_sampling.SVMSMOTE(sampling_strategy={1:860, 0:600})
# feature_src, src_label = sm.fit_resample(feature_src, src_label)
# print('Resampled dataset shape %s' % Counter(src_label))


from sklearn.linear_model import Lasso,LassoCV
"""       Lasso           400 features 0.000005   """

lasso = Lasso(alpha = 0.0005)
lasso.fit(feature_src, src_label)
coef = lasso.coef_
coef.astype('float64')
pos = np.where(abs(coef) > 0.000001)
pos = np.array(pos)
pos = pos[0,:]
print(len(pos))

feature_src = feature_src[:,pos]
feature_tar = feature_tar[:,pos]

# feature_src = coral(feature_src, feature_tar)

lr = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=500,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

# lr = LogisticRegression()

lr.fit(feature_src,src_label)
y_predict = lr.predict(feature_tar)
print("So, Our accuracy Score is: {}".format(round(accuracy_score(y_predict, tar_label), 4)))
print(confusion_matrix(tar_label, y_predict))

Y_score = lr.predict_proba(feature_tar)
Y_score = Y_score[:, 1]

FPR, TPR, _ = roc_curve(tar_label, Y_score)
ROC_AUC = auc(FPR, TPR)
print("Our AUC Score is: {}".format(round(ROC_AUC, 4)))
conf_matrix = confusion_matrix(tar_label, y_predict)
print('Calculating additional test metrics...')
accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1-Score: {}".format(f1_score))
