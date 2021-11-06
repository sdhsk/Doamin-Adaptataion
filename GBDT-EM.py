# 1.导入数据集

import xlrd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from xlrd import xldate_as_tuple
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV,KFold,train_test_split,cross_val_score
from sklearn.metrics import make_scorer , accuracy_score,r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import warnings
import random

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
# 1.导入数据集
warnings.filterwarnings("ignore")
np.random.seed(0)#产生随机种子意味着每次运行实验，产生的随机数都是相同的
file_location = r'C:\Users\Lenovo\Desktop\EM填充.xlsx'
data = xlrd.open_workbook(file_location)

sheet = data.sheet_by_index(0)
# print(sheet.cell_value(0,0))
# print(sheet.nrows-1)
# print(sheet.ncols)
# print("303行的值")
tablesl1=[]
for row in range(sheet.nrows):
    list1=[]
    for col in range(sheet.ncols-1):
        # print(sheet.cell_value(row,col))
        list1.append(sheet.cell_value(row,col+1))
    tablesl1.append(list1)
# print(tablesl1)
# print("27列的值")
tablesl2=[]
for col in range(sheet.ncols):
    list2=[]
    for row in range(sheet.nrows):
        # print(sheet.cell_value(row,col))
        list2.append(sheet.cell_value(row, col))
    tablesl2.append(list2)
# print(tablesl2[2])

print("自变量：'注射次数', '总药量', '日剂量', '性别', '年龄', '体重', '身高', 'BMI', 'cr μmol/L', '肌酐清除率（30以下、30-60、60-90、90以上四档）','女性肌酐清除率', '体温', '白细胞计数（*10^9）', '降钙素原水平(ng/ml)', 'APACHE II', '心脏疾病', '肺部感染', '糖尿病', '消化系统', '脑梗死', '泌尿系统','神经系统'")
ls1=[]
for i in range(1,len(tablesl1)):
    x=tablesl1[i]
    ls1.append(x)
print(ls1)

print("因变量:谷浓度")
ls2=[]
for i in range(1,len(tablesl2[0])):
    y=tablesl2[0][i]
    ls2.append(y)
print(ls2)


#2.将数据分割成训练集和测试集
from sklearn.model_selection import train_test_split
#随机采样25%
X_train, X_test, y_train, y_test = train_test_split(ls1, ls2, test_size=0.3)


# 特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
x_all=sc.fit_transform(ls1)
#https://zhuanlan.zhihu.com/p/81016622?from_voters_page=true



print("第1步：先来调n_estimators")
param_test1 = {'n_estimators':range(100,20000,100)}
gbdt =  GradientBoostingRegressor(loss='ls', learning_rate=0.001, subsample=1
                               , max_depth=19,min_samples_split=4,max_features=9
                                 , init=None, random_state=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False)

gsearch1 = GridSearchCV(estimator = gbdt,param_grid = param_test1, scoring='neg_mean_squared_error',iid=False,cv=10)
gsearch1.fit(X_train, y_train)
# print(gsearch1)
print(gsearch1.best_params_, gsearch1.best_score_)
print("第2步：一起调整max_depth和min_samples_split，根据输出的最优值将max_depth定下俩，后续再调整最小划分样本数")
# param_test2 = {'max_depth':range(1,30,2), 'min_samples_split':range(10,100,2)}
# gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, subsample=1,n_estimators=59
#                                 ,min_samples_leaf=1
#                                  , init=None, random_state=None, max_features=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False)
# gsearch2 = GridSearchCV(estimator = gbdt,param_grid = param_test2, scoring='neg_mean_squared_error',iid=False,cv=10)
# gsearch2.fit(X_train, y_train)
# print(gsearch2)
# print(gsearch2.best_params_, gsearch2.best_score_)
print("第3步：最小样本数min_samples_split64和叶子节点最少样本数min_samples_leaf一起调参")
# param_test3 = {'min_samples_split':range(10,100,2),'min_samples_leaf':range(1,50,1)}
# gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, subsample=1,n_estimators=59
#                                , max_depth=11,min_samples_split=26,min_samples_leaf=31
#                                  , init=None, random_state=None, max_features=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False)
# gsearch3 = GridSearchCV(estimator = gbdt,param_grid = param_test3, scoring='neg_mean_squared_error',iid=False,cv=10)
# gsearch3.fit(X_train, y_train)
# # print(gsearch2)
# print(gsearch3.best_params_, gsearch3.best_score_)
# param_test3 = {'min_samples_split':range(26,64)}63
# gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, subsample=1,n_estimators=59
#                                , max_depth=11,min_samples_split=26,min_samples_leaf=31
#                                  , init=None, random_state=None, max_features=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False)
# gsearch3 = GridSearchCV(estimator = gbdt,param_grid = param_test3, scoring='neg_mean_squared_error',iid=False,cv=10)
# gsearch3.fit(X_train, y_train)
# # print(gsearch2)
# print(gsearch3.best_params_, gsearch3.best_score_)
print("第4步：对最大特征数max_features进行网格搜索9")
# param_test4 = {'max_features':range(1,20,2)}
# gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, subsample=1,n_estimators=59
#                                , max_depth=11,min_samples_split=63,min_samples_leaf=31
#                                  , init=None, random_state=None, max_features=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False)
# gsearch4 = GridSearchCV(estimator = gbdt,param_grid = param_test4, scoring='neg_mean_squared_error',iid=False,cv=10)
# gsearch4.fit(X_train, y_train)
# # print(gsearch2)
# print(gsearch4.best_params_, gsearch4.best_score_)
print("第5步：对子采样的比例进行网格搜索")
# param_test5 = {'subsample':[0.6,0.69,0.7,0.71,0.72,0.73,0.75,0.8,0.85,0.9]}
# gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.1, subsample=1,n_estimators=59
#                                , max_depth=11,min_samples_split=26,max_features=9
#                                  , init=None, random_state=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False)
# gsearch5 = GridSearchCV(estimator = gbdt,param_grid = param_test5, scoring='neg_mean_squared_error',iid=False,cv=10)
# gsearch5.fit(X_train, y_train)
# # print(gsearch2)
# print(gsearch5.best_params_, gsearch5.best_score_)

#0.40421194966663637
# gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.6, subsample=1,n_estimators=30
#                                  , min_samples_split=100, min_samples_leaf=60, max_depth=3
#                                  , init=None, random_state=None, max_features=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False)
# gbdt.fit(X_train, y_train)
# print(np.average(abs(gbdt.predict(x_all)-y.ravel())))
#
# gbdt1 = GradientBoostingRegressor(loss='ls', learning_rate=0.001, subsample=1,n_estimators=5900
#                                , max_depth=11,min_samples_split=26,max_features=9
#                                  , init=None, random_state=None
#                                  , alpha=0.9, verbose=0, max_leaf_nodes=None
#                                  , warm_start=False)
# gbdt1.fit(X_train, y_train)
# # print(gbdt1.predict(x_all))
# #0.29835693480093495
# print(np.average(abs(gbdt1.predict(x_all)-y.ravel())))
#
gbdt2 = GradientBoostingRegressor(loss='ls', learning_rate=0.001, subsample=1,n_estimators=6000
                               , max_depth=19,min_samples_split=4,max_features=9
                                 , init=None, random_state=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False)
gbdt2.fit(X_train, y_train)
# print(gbdt2.predict(x_all))
# print("gbdt2_mae",np.average(abs(gbdt2.predict(x_all)-ls2)))
#
gbdt3 = GradientBoostingRegressor(loss='ls', learning_rate=0.001, subsample=1,n_estimators=10000
                               , max_depth=15,min_samples_split=2,max_features=19
                                 , init=None, random_state=None
                                 , alpha=0.9, verbose=1, max_leaf_nodes=None
                                 , warm_start=False)
gbdt3.fit(X_train, y_train)
print(gbdt3.predict(x_all))
print("gbtd3_mae",np.average(abs(gbdt3.predict(x_all)-ls2)))

gbdt4 = GradientBoostingRegressor(loss='ls', learning_rate=0.00001, subsample=1,n_estimators=6000
                               , max_depth=15,min_samples_split=2,max_features=19
                                 , init=None, random_state=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False)
gbdt4.fit(X_train, y_train)
# print(gbdt4.predict(x_all))
# print("gbtd4_mae",np.average(abs(gbdt4.predict(x_all)-ls2)))

gbdt5 = GradientBoostingRegressor(loss='ls', learning_rate=0.001, subsample=1,n_estimators=6000
                               , max_depth=19,min_samples_split=2,max_features=19
                                 , init=None, random_state=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False)

gbdt5.fit(X_train, y_train)
pre5=gbdt5.predict(x_all)
# print('gbdt5_mae',np.average(abs(pre5-ls2)))
plt.plot(ls2)
plt.plot(gbdt5.predict(x_all))
plt.show()



gbdt6 = GradientBoostingRegressor(loss='ls', learning_rate=0.001, subsample=1,n_estimators=6000
                               , max_depth=20,min_samples_split=2,max_features=20
                                 , init=None, random_state=None
                                 , alpha=0.9, verbose=0, max_leaf_nodes=None
                                 , warm_start=False)
gbdt6.fit(X_train, y_train)
pre6=gbdt6.predict(x_all)
# print('gbdt6_mae',np.average(abs(pre6-ls2)))
plt.plot(ls2)
plt.plot(gbdt6.predict(x_all))
plt.show()

#
# from six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
# import os
#
# # 执行一次
# from sklearn import tree
#
# import pydot
# sub_tree_42 = gbdt5.estimators_[0, 0]
# dot_data= tree.export_graphviz(sub_tree_42, out_file=None)
# graph = pydot.graph_from_dot_data(dot_data)
# print(graph[0])
#
# dot_data = StringIO()
# tree.export_graphviz(gbdt5.estimators_[0,0],out_file = dot_data,node_ids=True,filled=True,rounded=True,special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png("gbdt.png")
# graph.write_pdf("gbdt.pdf")
# print('Visible tree plot saved as pdf.')
#
#
#
#
#
#
