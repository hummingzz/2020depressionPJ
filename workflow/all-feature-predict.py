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
from CFSmethod import CFS


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

audio_data_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/audiofeaturespace'
deep_data_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/fullspace'
attention_deep_data_path = '/home/pehuang/zhaozhang/beijing/data07_tf/npytotal/fullspace_1'
wang_data_path = '/home/pehuang/zhaozhang/beijing/data07_tf/data_wang'

data_train = np.arange(1*568).reshape(1,568)
data_test = np.arange(1*568).reshape(1,568)

for file in os.listdir(audio_data_path):
    if file not in ['data_full_model_07_new.csv']:
        judgement = file[-6:-4]
        file_path = os.path.join(audio_data_path,file)
        audio_feature = pd.read_csv(file_path)
        # for file in os.listdir(deep_data_path):
        for file in os.listdir(attention_deep_data_path):
            if file[:2]  == judgement:
                file_path = os.path.join(attention_deep_data_path, file)
                # file_path = os.path.join(deep_data_path,file)
                deep_feature = pd.read_csv(file_path)
                deep_feature.drop(columns=['TYPE','BDI','AGE'],axis=1,inplace=True)
                feature = pd.merge(audio_feature,deep_feature,on='partic_id')
                # feature = pd.concat((audio_feature,deep_feature),axis=1)
                for file in os.listdir(wang_data_path):
                    if file[-10:-8] == judgement:
                        file_path = os.path.join(wang_data_path, file)
                        wang_feature = pd.read_csv(file_path)
                        wang_feature.drop(columns=['TYPE','BDI','SEX','AGE','silence_len','silence_sum','silence_ave'],axis=1,inplace=True)
                        feature = pd.merge(feature, wang_feature, on='partic_id')
                        print(feature.head())
                        print(feature.shape)
                        feature_for_train = pd.merge(feature, train_id['partic_id'], how='inner', on='partic_id')
                        feature_for_test = pd.merge(feature, test_id['partic_id'], how='inner', on='partic_id')
                        print(feature_for_train.shape)
                        print(feature_for_test.shape)
                        data_train = np.concatenate((data_train,feature_for_train.values), axis=0)
                        data_test = np.concatenate((data_test, feature_for_test.values), axis=0)
                        print(data_train.shape)
                        print(data_test.shape)



data_train = data_train[1:,:]
data_test = data_test[1:,:]
data_train = pd.DataFrame(data_train, columns=feature_for_train.columns)
data_test = pd.DataFrame(data_test, columns=feature_for_test.columns)


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

train_data = pd.get_dummies(train_data, columns=['SEX'])
test_data = pd.get_dummies(test_data, columns=['SEX'])

column_0 = list(train_data.columns)
print(len(column_0))
column_0.remove('4')
column_0.remove('36_x')
print(len(column_0))



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

print(feature_src.shape)
print(feature_tar.shape)
print(src_label.shape)
print(tar_label.shape)
input()


# print('Before resampled dataset shape %s' % Counter(src_label))
# sm = over_sampling.SVMSMOTE(sampling_strategy={1:1400, 0:1000})
# sm = over_sampling.SVMSMOTE(sampling_strategy={1:720, 0:500})
# feature_src, src_label = sm.fit_resample(feature_src, src_label)
# print('Resampled dataset shape %s' % Counter(src_label))


from sklearn.linear_model import Lasso,LassoCV


# for j in np.arange(0, 3, 0.00005):
#     model_lasso = LassoCV(alphas = [j]).fit(feature_src, src_label)
#     coef = model_lasso.coef_
#     pos = np.where(abs(coef)>0.000001)
#     pos = np.array(pos)
#
#     pos = pos[0,:]
#
#     if 300 < len(pos) < 400 :
#         print(len(pos))
#         break

#model2 = Lasso(alpha=.3)
#model2.fit(X_train, y_train)
#coef=model_lasso.coef_
#pos=np.where(abs(coef)>0.0001)
#pos=np.array(pos)
#pos=pos[0,:]
#dd=coef[pos]
#ee=abs(dd)
#ff=ee.argsort()
#ff=list(ff)
#max_num_index_list = map(ff.index, heapq.nlargest(3, ff))
#max_num_index_list=list(max_num_index_list)
#posfor=pos[max_num_index_list]
# selected_feature=X_train[:,pos]
# selected_feature_test=X_test[:,pos]


"""    we need    """
# lasso = Lasso(alpha = 0.00005)
# lasso.fit(feature_src, src_label)
# coef = lasso.coef_
# coef.astype('float64')
# pos = np.where(abs(coef) > 0.0000001)
# pos = np.array(pos)
# pos = pos[0,:]
# print(len(pos))

# feature_src = feature_src[:,pos]
# feature_tar = feature_tar[:,pos]
print(feature_src[:,39])
print(feature_src[:,543])

feature_src = np.delete(feature_src, [39,543], axis=1)
feature_tar = np.delete(feature_tar, [39,543], axis=1)



feature_src = pd.DataFrame(data=feature_src,columns=column_0)
feature_tar = pd.DataFrame(data=feature_tar,columns=column_0)
src_label = pd.DataFrame(data=src_label, columns=['label'])
tar_label = pd.DataFrame(data=tar_label, columns=['label'])

print(feature_src.shape)
print(feature_tar.shape)
print(src_label.shape)
print(tar_label.shape)
# feature_src.to_csv('/home/pehuang/zhaozhang/beijing/wangfeatureselection/files/feature_src.csv',index=None)
# src_label.to_csv('/home/pehuang/zhaozhang/beijing/wangfeatureselection/files/src_label.csv',index=None)
# feature_tar.to_csv('/home/pehuang/zhaozhang/beijing/wangfeatureselection/files/feature_tar.csv',index=None)
# tar_label.to_csv('/home/pehuang/zhaozhang/beijing/wangfeatureselection/files/tar_label.csv',index=None)
# input()

# feature_src.drop(['36_x','4'], axis=1, inplace=True)
# feature_tar.drop(['36_x','4'], axis=1, inplace=True)
# column_1 = feature_src.columns
# print(feature_src.shape)
# print(feature_tar.shape)

# feature_src = feature_src.drop(['36_x'],axis=1)

# column_1 = feature_src.columns
# feature_src = feature_src.values[:,:100]
# src_label = src_label.values


"""   save correlation   """
# data = pd.concat([feature_src,src_label],axis=1)
# cor = data.corr()
# cor.to_csv('/home/pehuang/zhaozhang/beijing/workflowresult/corr.csv',index=None)
#
#
#
# """   save feature and label   """
# feature_src.to_csv('/home/pehuang/zhaozhang/beijing/workflowresult/feature.csv',index=None,encoding='gbk')
# print('finished')
# input()

# src_label.to_csv('/home/pehuang/zhaozhang/beijing/workflowresult/target.csv',index=None,encoding='gbk')
# feature_src[['10','37_y','117_y','98_y','36_x','2_y']].to_csv('/home/pehuang/zhaozhang/beijing/spec1/data_train.csv', index=None)
# feature_tar[['10','37_y','117_y','98_y','36_x','2_y']].to_csv('/home/pehuang/zhaozhang/beijing/spec1/data_test.csv', index=None)
# src_label.to_csv('/home/pehuang/zhaozhang/beijing/spec1/label_train.csv', index=None)
# tar_label.to_csv('/home/pehuang/zhaozhang/beijing/spec1/label_test.csv', index=None)
# print(feature_src.shape)
# print(feature_src.head())

# feature_src = np.delete(feature_src, 39, axis=1)
# feature_tar = np.delete(feature_tar, 39, axis=1)

# feature_src = np.array(feature_src, dtype=np.float)
# src_label = np.array(src_label, dtype=np.float)

# idx = CFS.cfs(feature_src, src_label)
#
# selected_idx = column_0[idx]
# print(idx)
# print(selected_idx)
#
# input()

# first idx = [549,449,529,510,39,414]

# drop 0 hou de idx = [548,448,528,509,542,413]

# feature_src = feature_src[:,idx]
# feature_tar = feature_tar[:,idx]
# print(feature_src.shape)
# print(feature_tar.shape)


# feature_tar = coral(feature_tar,feature_src)
# feature_src = coral(feature_src, feature_tar)

"""   Model Parameter Selection    """
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

n_estimators = [50,100, 200, 500]
n_learning = [0.3, 0.1, 0.01, 0.001]
n_max_depth = [3, 5, 8, 10, 20]
n_gamma = [0, 0.1, 0.3, 0.5, 1]
n_colsample = [0.6 , 0.8, 1, 0.5]
n_minchild = [1, 2, 3]
n_criterion = ['gini', 'entropy']
n_features = ['sqrt', 'log2', 0.8, 0.6]
n_leaf = [2, 5, 10]
n_split = [2, 5, 10]
cv = StratifiedShuffleSplit(n_splits=5, test_size=.30, random_state=22)

parameters = {'n_estimators': n_estimators,
              'criterion' : n_criterion,
              'max_features' : n_features,
              'min_samples_leaf' : n_leaf,
              'min_samples_split' : n_split,
              'max_depth': n_max_depth
              }

# grid = GridSearchCV(estimator=RandomForestClassifier(),
#                     param_grid=parameters,
#                     scoring='roc_auc',
#                     cv=cv,
#                     n_jobs=-1)

# grid.fit(feature_src, src_label)
# print(grid.best_score_)
# print(grid.best_params_)
# print(grid.best_estimator_)
# rf_grid = grid.best_estimator_
# print('---------------')
# print(rf_grid.score(feature_tar,tar_label))

import stepwiseSelection as ss
import statsmodels.formula.api as sm
import statsmodels.api as sm_1

# feature_src.drop(['4'], axis=1)
# # src_label = src_label.ix[:200,:]
# # feature_src = feature_src.ix[:200,:]

# final_vars, iterations_logs = ss.backwardSelection(feature_src,src_label, model_type="linear")
# print(len(final_vars))
# feature_src = feature_src[final_vars[1:]]
# feature_tar = feature_tar[final_vars[1:]]
# print(feature_src.shape)

# final_vars = np.array(final_vars)
# print(final_vars)

lr = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=8, max_features=0.6, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
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
