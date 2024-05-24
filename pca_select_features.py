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
dim_re = PCA(n_components=2)#PCA()
#dim_re= LinearDiscriminantAnalysis(n_components=3)
#dim_re= TSNE(n_components=3)#, , n_iter=300
#dim_re = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=10)

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



columns_train2=['FAM_ID','IND_ID','LIN_ID']


#read space separated values file with pandas

df_train=pd.read_csv('Full_annotate_dataset.csv')



print(df_train.shape)



print(df_train.head())










df_train2=pd.read_csv('Family_Codes_Full_Dataset.csv', sep='\t')  





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
#le= preprocessing.LabelEncoder()
#y_train = le.fit_transform(df_train2.iloc[: , 2]) 
 
print(",,,,",y_train.shape)


df2=df_train.iloc[:, 1:]

print("===^^^^",df2.head())

print("===^^^^",df2.shape)


'''
#implement feature reduction algorithms 
x_pca = dim_re.fit_transform(df2)
df2 = pd.DataFrame(x_pca)


print("$$$^^^^",df2.head())

print("$$$^^^^",df2.shape)
'''


df_train=df2



###########

from sklearn import decomposition
from sklearn.preprocessing import scale

X = scale(df2)
y = y_train
print(y_train.shape)

# apply PCA
pca1 = decomposition.PCA(n_components=2)
X = pca1.fit_transform(X)


print("YYYYY",type(df2.columns))
loading_matrix = pd.DataFrame(pca1.components_.T, columns=['PC1', 'PC2'], index=df2.columns) #df2.columns)#index=df.loc[:,::2] 
print("loading_matrix",loading_matrix)
print(loading_matrix.shape)


df = loading_matrix

print(df['PC1'])

df['PC1']=df['PC1'].abs()

df=df.drop_duplicates(subset=['PC1'])

print("No duplicates",df.shape)

df=df.nlargest(n=96, columns=['PC1'])

print("wwwww",df)



feature = list(df.index)# https://www.geeksforgeeks.org/how-to-get-rows-index-names-in-pandas-dataframe/
print("no of Selected_SNPs=",len(feature))

print(type(feature))

df = pd.DataFrame(feature, columns =['SNPs_ids'])   #To pickup the minimum no of SNPs
df.to_csv('96_slected_features_PCA.csv',index=False) 



