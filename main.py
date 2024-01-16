import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


#from keras.utils import np_utils
#from keras.layers import Dense, Conv2D, MaxPooling2D
#from keras.layers import Dropout, Flatten, GlobalAveragePooling2D


# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC

#for dimensionality reduction to 10 features

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
#dim_re = PCA(n_components=10)#PCA()
#dim_re= LinearDiscriminantAnalysis(n_components=3)
#dim_re= TSNE(n_components=3)#, , n_iter=300
dim_re = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=10)

import itertools



 
# Importing hypopt library for grid search
#from hypopt import GridSearch

import random
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


from imblearn.over_sampling import SMOTE  #to import Synthetic Minority Over-sampling Technique (SMOTE) algorithm
sm = SMOTE(random_state=42)

oversample = SMOTE()

#######################################################
  
# training a Classifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

#clf = GaussianNB()
#clf=LogisticRegression()#C=1e4
#clf=LogisticRegression(random_state=random.seed(1234))
#clf=DecisionTreeClassifier()
#clf=KNeighborsClassifier()
#clf=KNeighborsClassifier(n_neighbors=2)#(n_neighbors=3)default#n_neighbors=5
#clf=svm.SVC()

weights = {0:1.0, 1:1.4, 2:8.0, 3:16.0, 4:16.0}
#clf = RandomForestClassifier()
########clf = RandomForestClassifier(n_estimators=1000, class_weight=weights)
#####clf=RandomForestClassifier(n_estimators=30, max_depth=10, random_state = random.seed(1234))#,class_weight=weights)#random_state=0)
#clf = MLPClassifier(random_state=1, max_iter=300)
#clf = MLPClassifier()

#####################
from sklearn.utils import class_weight
#from sklearn.utils.class_weight import compute_class_weight
#####################

import xgboost as xgb
clf= xgb.XGBClassifier()

'''
clf= xgb.XGBClassifier(colsample_bytree= 1,
 eta= 0.01,
 gamma= 0.1,
 max_depth= 15,
 min_child_weight= 3,
 scale_pos_weight= 1,
 silent= True)
'''

print(clf)

import warnings
warnings.filterwarnings('ignore')






##############################################################

columns_train=['rs#','alleles','chrom',	'pos','strand','assembly#','center','protLSID','assayLSID','panelLSID','QCcode']

columns_train2=['FAM_ID','IND_ID','LIN_ID']


#read space separated values file with pandas
df_train=pd.read_csv('Full_dataset12.ped', delimiter=' ')  

#df_train=pd.read_csv('Dataset_500_no_update.ped', sep='\t') 

print(df_train.shape)

#To drop column 5 with -9 and N/A values #phenotype

df_train = df_train.drop(df_train.columns[[5]],axis = 1)

print(df_train.shape)

print(df_train.head())










df_train2=pd.read_csv('Family_Codes_Full_Dataset.csv', sep='\t', skiprows=1)  





############################

############################
print("))))", df_train.shape)



#############################################################


print(",,,,",df_train2.shape)



y_train = df_train2.iloc[: , 2]
print(",,,,",y_train.shape)
print(y_train.head())


#label encoding of y four breeds groups.
from sklearn import preprocessing  
le= preprocessing.LabelEncoder()
y_train = le.fit_transform(df_train2.iloc[: , 2]) 
 
print(",,,,",y_train.shape)

df1=df_train.iloc[:, 0:5]
print("$$$^^^^",df1.head())
df2=df_train.iloc[:, 5:]#.sample(n=100, axis=1)


#implement feature reduction algorithms 
#x_pca = dim_re.fit_transform(df2)
#x_lda = dim_re.fit(df2, y_train).transform(df2)
#x_tsne = TSNE(learning_rate=100).fit_transform(df2)
X_umap=dim_re.fit_transform(df2)

#df2 = pd.DataFrame(x_pca)

#df2 = pd.DataFrame(x_lda)

#df2 = pd.DataFrame(x_tsne)

df2 = pd.DataFrame(X_umap)

print("$$$^^^^",df2.head())

print("$$$^^^^",df2.shape)



#df_train = pd.concat([df1, df2], axis=1)



df_train=df2

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_train.iloc[:, 0] = le.fit_transform(df_train.iloc[:, 0])
df_train.iloc[:, 1] = le.fit_transform(df_train.iloc[:, 1])

print(df_train.iloc[:, 0:2].head())
'''

#implement feature selection algorithms with RF

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1, max_depth=10)
feature = SelectFromModel(model)
df_train= feature.fit_transform(df_train, y_train)

print(df_train.shape)

print("////",type(df_train))
features_train = df_train

'''
features_train = df_train.to_numpy()
#features_train = df_train[columns_train].to_numpy()

print("^^^^",features_train.shape)


#################################################################

#features_train=lesions_features_train

print("validation_part with ML")

X_train=features_train


#X_train= scaler.fit_transform(X_train)


from sklearn.model_selection import train_test_split 

#dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state = 0) 


print(type(X_train))
print(type(y_train))

print("@@@",X_train.shape)
print("@@@",y_train.shape)

#model = clf.fit(X_train, y_train)


#from imblearn.over_sampling import SMOTE
#sm = SMOTE(random_state=42)

#SMOTE
#X_train, y_train= sm.fit_resample(X_train, y_train)



#model = clf.fit(X_train, y_train)

from sklearn.utils.class_weight import compute_sample_weight

model = clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))


print("SSSSSSSSSSSSS",X_test)

###end of lesion part

y_pred  = model.predict(X_test)
print(type(y_pred)) #ndarray

print("*****",y_pred.shape)

print("@@@",y_pred)
print("@@@",y_test)
##########################################

# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix
# creating a confusion matrix 
#cm = confusion_matrix(y_test, y_pred )
print("y_pred.shape=",y_pred.shape)
print(y_pred[0: 5])
print("y_test.shape=",y_test.shape)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
 
print(classification_report(y_test, y_pred))
#auc=roc_auc_score(y_test.round(),y_pred,multi_class="ovr",average=None)
#auc = float("{0:.3f}".format(auc))
#print("AUC=",auc)

#true negatives c00, false negatives C10, true positives C11, and false positives C01 
#tn c00, fpC01, fnC10, tpC11 
l=confusion_matrix(y_test, y_pred)#https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
print(l)

print(clf)
print(dim_re)
#print(columns_train)


