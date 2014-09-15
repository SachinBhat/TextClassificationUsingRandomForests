
import cPickle
import numpy as NP
import matplotlib.pyplot as PLT
from scipy import interp

import time, sys, csv

from sklearn import preprocessing as SK_PP
from sklearn import svm as SK_SVM
from sklearn import naive_bayes as SK_NB
from sklearn import tree as SK_TR
from sklearn import ensemble as SK_EN
from sklearn import linear_model as SK_LM
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.externals import joblib

dir1 = "../data_sachin/"
TftFile = ''.join([dir1,'Features_train2_az.csv'])
#CftFile = ''.join([dir1,'train_cat1_features_az.csv'])
lblFile = ''.join([dir1,'train_label_vector_az.csv'])
sclFile = ''.join([dir1,'scaler2_az.pkl'])
clsfFile = ''.join([dir1,'classifier_RandomForest2_az.pkl'])


print "Loading text features..."
sys.stdout.flush()

X_train = NP.loadtxt(TftFile,delimiter=",",skiprows=1)
print " train data text features read"
sys.stdout.flush()

print "Loading category features..."
sys.stdout.flush()

##C_features = NP.loadtxt(CftFile,delimiter=",",skiprows=1)
##print " train data category features read"
##sys.stdout.flush()

print "Loading output labels..."
sys.stdout.flush()

y_train = NP.loadtxt(lblFile,delimiter=",",skiprows=1)
print " train labels read"
sys.stdout.flush()

#X_train = NP.concatenate((T_features,''),axis=1)

# Scale the features
print "Scaling..." 
sys.stdout.flush()

#with open('scaler.pkl', 'r') as fid:
#    scaler_train = cPickle.load( fid) 

scaler_train = SK_PP.Scaler().fit(X_train)
X_train_scaled = scaler_train.transform(X_train)


with open(sclFile, 'wb') as fid:
    cPickle.dump(scaler_train, fid) 



classifier=SK_EN.RandomForestClassifier(n_estimators=200, min_samples_split=50 ,min_samples_leaf = 50, verbose=1,  compute_importances=True)
print "Classifying with... Random Forests"
sys.stdout.flush()

classifier5 = classifier.fit(X_train_scaled, y_train)
with open(clsfFile, 'wb') as fid:
    cPickle.dump(classifier5, fid) 


