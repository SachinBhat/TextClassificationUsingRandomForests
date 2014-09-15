
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
#CftFile = ''.join([dir1,'test_cat1_features_az.csv'])
sclFile = ''.join([dir1,'scaler2_az.pkl'])
clsfFile = ''.join([dir1,'classifier_RandomForest2_az.pkl'])
tprFile = ''.join([dir1,'test_predicted_RandomForest2_az_v2.csv'])

outWriter_fp1 = open(tprFile, 'wb')
outWriter1 = csv.writer(outWriter_fp1)
process_line1 = []
print "Loading text features..."
sys.stdout.flush()

X_test = NP.loadtxt(TftFile,delimiter=",",skiprows=1)
print "test data text features read"
sys.stdout.flush()

print "Loading category features..."
sys.stdout.flush()

##C_features = NP.loadtxt(CftFile,delimiter=",",skiprows=1)
##print "test data category features read"
##sys.stdout.flush()


#X_test = NP.concatenate((t_features,''),axis=1)


# Scale the features
print "Scaling..." 
sys.stdout.flush()
with open(sclFile, 'r') as fid:
    scaler_train = cPickle.load( fid) 

X_test_scaled = scaler_train.transform(X_test)

print "Testing with... Random Forest"

with open(clsfFile, 'r') as fid:
    classifier2 = cPickle.load(fid) 

probas_2 = classifier2.predict_proba(X_test_scaled)

a = NP.asarray(probas_2)
b = NP.column_stack((a))
print b
print( "shape --> %d rows x %d columns" % b.shape )
for i in b:
    process_line1 = []
    process_line1.append(i)
    outWriter1.writerow(process_line1)
#NP.savetxt(tprFile, b, delimiter=",")

outWriter_fp1.close()

