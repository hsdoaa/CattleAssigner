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
clf = RandomForestClassifier()
########clf = RandomForestClassifier(n_estimators=1000, class_weight=weights)
#####clf=RandomForestClassifier(n_estimators=30, max_depth=10, random_state = random.seed(1234))#,class_weight=weights)#random_state=0)
#clf = MLPClassifier(random_state=1, max_iter=300)
#clf = MLPClassifier()

#####################
from sklearn.utils import class_weight
#from sklearn.utils.class_weight import compute_class_weight
#####################

import xgboost as xgb
#clf= xgb.XGBClassifier()

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
print("))))", df_train.shape)

#############################################################


print(",,,,",df_train2.shape)



y_train = df_train2.iloc[: , 2]
print(",,,,",y_train.shape)
print(y_train.head())


#label encoding of y four breeds groups.
from sklearn import preprocessing  
le1= preprocessing.LabelEncoder()
y_train = le1.fit_transform(df_train2.iloc[: , 2]) 
 
print(",,,,",y_train.shape)


df2=df_train.iloc[:, 1:]

print("===^^^^",df2.head())

print("===^^^^",df2.shape)




df_train=df2



###########

features_train = df_train

#features_train = df_train.to_numpy()

print("^^^^",features_train.shape)





#################################################################


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




from sklearn.utils.class_weight import compute_sample_weight
rf = RandomForestClassifier()#(n_estimators=100)
model = rf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
from sklearn.inspection import permutation_importance
importance = permutation_importance(model, X_test, y_test)

importance = importance.importances_mean

print("type of importance=",type(importance)) #numpy array
#print(importance.shape)



SNP_names=[]
Scores=[]
for i,v in enumerate(importance):
 SNP_names.append(i)
 Scores.append(v) 

print(type(SNP_names))
print(type(Scores))
  
df = pd.DataFrame({'SNP_name':SNP_names,'Score':Scores})

print(df.shape)
print(df.info())
print(df.head())

print(df['Score'])

df['Score']=df['Score'].abs()

#df=df.drop_duplicates(subset=['Score'])

#print("No duplicates",df.shape)

df=df.nlargest(n=192, columns=['Score'])

print("wwwww",df)


feature = list(df.index)# https://www.geeksforgeeks.org/how-to-get-rows-index-names-in-pandas-dataframe/
print("no of Selected_SNPs=",len(feature))

##### end of feature selection

#################################################################
print(features_train.shape)
#df.iloc[:, 3]
X=features_train.iloc[:,feature]  #to train the classifer with the selected features
X.to_csv('192_slected_features_RF_mda.csv',index=False) 

