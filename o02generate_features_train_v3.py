# generate_features.py 

import sys, csv, codecs, re
from math import log
from sklearn.externals import joblib
import cPickle

dir1 = "../data_sachin/"
dir2 = "../data_sachin/"
posFile = ''.join([dir1,'train_data_az.csv'])
taxFile = ''.join([dir2,'taxAdict2_az.pkl'])
prepFile = ''.join([dir2,'prep2_az.pickle'])
outFile = ''.join([dir2,'Features_train2_az.csv'])
outFile2 = ''.join([dir2,'IDF_train2_az.csv'])


### Develop a corpus of words (Go through all available text to find all the possible words)
print "Gathering word corpus..."
sys.stdout.flush()
aOverallDist = {}
aNumDocHits  = {}
totalNumDocs = 0
for label in [1]:
    if label: dataFile = csv.reader(open(posFile, 'rb'))

    header = dataFile.next()
    print header
    fields = dict(zip(header,range(len(header))))
    for item in dataFile:
        try:
            text = item[fields['SUMMARY']].decode('utf-8')
        except UnicodeDecodeError:
            print "UnicodeDecodeError on entry:",item[fields['CASE_ID_']]
            sys.exit(0)

        totalNumDocs += 1
        if totalNumDocs%1000==0:
            print totalNumDocs
            sys.stdout.flush()

        for word in list(set(text.split(" "))):
            word = word.strip()
            # remove "common" words, aka stopwords
            if word==" " or word=="": continue
            if word not in aOverallDist:
                aOverallDist[word]=0
                aNumDocHits[word]=0
            # Overall Counts
            aOverallDist[word] += text.count(word)

            # Document Frequency for each Word
            aNumDocHits[word] += 1

### This is our corpus. Optionally, we could have imported this corpus from elsewhere.
### If the dataset is too large, we can increase the threshold for number of times a word appears
taxAdict = [word for word in aNumDocHits if aNumDocHits[word]>50]
# taxAdict = [word for word in aOverallDist if aOverallDist[word]>50]      
print str(len(taxAdict)), "words."

with open(taxFile, 'wb') as fid:
    cPickle.dump(taxAdict, fid) 
    
    
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
    if label: dataFile = csv.reader(open(posFile, 'rb'))

    header = dataFile.next()
    fields = dict(zip(header,range(len(header))))
    for item in dataFile:
        try:
            text = item[fields['SUMMARY']].decode('utf-8')
        except UnicodeDecodeError:
            print "UnicodeDecodeError on entry:",item[fields['CASE_ID_']]
            sys.exit(0)
        counter = counter+1
        if counter%1000 ==0:
            print counter
            sys.stdout.flush()
        aRawWordExistenceDict = dict(zip(taxAdict,[0]*len(taxAdict)))

        text1 = text.lower().split(" ")
        for word in taxAdict:
            if word in text1:
            # Word Existence Matrix
            # Word Existence Matrix
                aRawWordExistenceDict[word]+= text.count(word)

        toWrite = []
        for word in sorted(aRawWordExistenceDict.keys()):
            toWrite.append(aRawWordExistenceDict[word])

        outWriter.writerow(toWrite)

outWriter_fp.close()

### Reprocess to remove features that are just all zero's 
print "Post-processing word count features..."
with open(outFile,'rb') as outReader_fp:
    outReader = csv.reader(outReader_fp)

    header = outReader.next()
    goodIndices = [i for i,word in enumerate(header) if aOverallDist[word.decode('utf-8')]>0]

    newHeader = [header[i] for i in goodIndices] # still encoded
    newOutFile = [[line[i] for i in goodIndices] for line in outReader]

with open(outFile,'wb') as outWriter_fp:
    outWriter = csv.writer(outWriter_fp)

    outWriter.writerow(newHeader)
    for line in newOutFile:
        outWriter.writerow(line)


### Reprocess to convert to TF-IDF (see Wikipedia)
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
            if i==0: continue
            DC = aNumDocHits[header[i].decode('utf-8')]
            #assert DC>0, "Document Frequency is zero... this should have been deleted"
            IDF = log(float(totalNumDocs)/float(DC))
            allIDFs[i]=IDF
            line[i] = float(TF)*IDF
        counter = counter+1
        if counter%1000==0:
            print counter
            sys.stdout.flush()

        newOutFile.append(line)
    newOutFile2.append(allIDFs)

with open(outFile,'wb') as outWriter_fp:
    outWriter = csv.writer(outWriter_fp)
    for line in newOutFile:
        outWriter.writerow(line)

with open(outFile2,'wb') as outWriter_fp2:
    outWriter2 = csv.writer(outWriter_fp2)
    for line in newOutFile2:
        outWriter2.writerow(line)

        
outWriter_fp.close()
outWriter_fp2.close()

### Dump the transformations to files for our model
cPickle.dump({'taxAdict':taxAdict, 'goodIndices':goodIndices,'allIDFs':allIDFs}, open(prepFile,'wb'),2)
