# generate_features.py 

import sys, csv, codecs, re, cPickle
from math import log
from sklearn.externals import joblib
import pickle

dir2 = "../data_sachin/"

posFile = ''.join([dir2,'test_data_az_v2.csv'])
taxFile = ''.join([dir2,'taxAdict2_az_v2.pkl'])
outFile = ''.join([dir2,'Features_test_outoftime_az_v2.csv'])
train_IDF = ''.join([dir2,'IDF_train2_az_v2.csv'])


### Develop a corpus of words (Go through all available text to find all the possible words)
print "Gathering word corpus...",

### Import training corpus ######################################################################
#################################################################################################

# taxAdict = [word for word in aOverallDist if aOverallDist[word]>50]
with open(taxFile, 'r') as fid:
    taxAdict = cPickle.load(fid) 
print str(len(taxAdict)), "words."

outWriter_fp = open(outFile, 'wb')
outWriter = csv.writer(outWriter_fp)

header = []
for word in sorted(taxAdict):
    header.append(word.encode('utf-8'))
outWriter.writerow(header)

counter =0
### Get Word Counts (Features) for each document
print "Getting Document Word Counts..."
for label in [1]:
    if label:
        dataFile = csv.reader(open(posFile, 'rb'))
        print "Test file read"

    header = dataFile.next()
    #print header
    fields = dict(zip(header,range(len(header))))
    #print fields
    for item in dataFile:
        try:
            text = item[fields['SUMMARY']].decode('utf-8')
        except UnicodeDecodeError:
            print "UnicodeDecodeError on entry:",item[fields['CASE_ID_']]
            sys.exit(0)
        counter = counter+1
        if counter%1000 ==0:
            print counter
            #print text
            sys.stdout.flush()
 
        aRawWordExistenceDict = dict(zip(taxAdict,[0]*len(taxAdict)))

        text1 = text.lower().split(" ")
        for word in taxAdict:
            if word in text1:
                # Word Existence Matrix
                aRawWordExistenceDict[word]+= text.count(word)

        toWrite = []
        for word in sorted(aRawWordExistenceDict.keys()):
            toWrite.append(aRawWordExistenceDict[word])


        outWriter.writerow(toWrite)

outWriter_fp.close()

### Reprocess to remove features that are just all zero's 
print "Post-processing word count features..."

### Reprocess to convert to TF-IDF (see Wikipedia)
#####################################################################        

with open(train_IDF, 'rb') as f:
    reader = csv.reader(f)
    counter = 0
    for row in reader:
        if counter ==1 :
            train_IDF2= row
        counter=counter+1
flag = 0

with open(outFile,'rb') as outReader_fp:
    outReader = csv.reader(outReader_fp)

    header = outReader.next()
    newOutFile = [header]
    newOutFile2 = [header]
    allIDFs = [0]*len(header)
    IDFs = [0]*len(header)
    counter=0
    for line in outReader:
        for i,TF in enumerate(line):
            #print header[i]
            #DC = aNumDocHits[header[i].decode('utf-8')]
            #assert DC>0, "Document Frequency is zero... this should have been deleted"
            IDF = float(train_IDF2[i])
            #print header[i],i,IDF
            #IDF = log(float(totalNumDocs)/float(DC))
            
            allIDFs[i]=IDF
            line[i] = float(TF)*IDF
        counter = counter+1
        if counter%1000==0:
            print counter
            sys.stdout.flush()
        newOutFile.append(line)

with open(outFile,'wb') as outWriter_fp:
    outWriter = csv.writer(outWriter_fp)
    for line in newOutFile:
        outWriter.writerow(line)

outWriter_fp.close()

