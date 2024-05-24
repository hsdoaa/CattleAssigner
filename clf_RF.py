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

from sklearn.ensemble import RandomForestClassifier

weights = {0:1.0, 1:1.4, 2:8.0, 3:16.0, 4:16.0}
clf = RandomForestClassifier()

#####################
from sklearn.utils import class_weight
#from sklearn.utils.class_weight import compute_class_weight
#####################

print(clf)

import warnings
warnings.filterwarnings('ignore')

##############################################################


#read space separated values file with pandas

df_train=pd.read_csv('192_slected_features_RF_mda.csv')

print(df_train.shape)

print(df_train.head())

#############################################################
df_train2=pd.read_csv('Family_Codes_Full_Dataset.csv', sep='\t')  

print(",,,,",df_train2.shape)



y = df_train2.iloc[: , 2]
print(",,,,",y.shape)
print(y.head())


#label encoding of y four breeds groups.
from sklearn import preprocessing  
le1= preprocessing.LabelEncoder()
y = le1.fit_transform(df_train2.iloc[: , 2]) 

##############################################################


###########

features_train = df_train

#features_train = df_train.to_numpy()

print("^^^^",features_train.shape)



################################################################
X=features_train

print("validation_part with ML")


#X_train= scaler.fit_transform(X_train)


from sklearn.model_selection import train_test_split 

#dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0) 


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


#print("SSSSSSSSSSSSS",X_test)


y_pred  = model.predict(X_test)
print(type(y_pred)) #ndarray

print("*****",y_pred.shape)

#print("@@@",y_pred)
#print("@@@",y_test)
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
#print(columns_train)





