# preprocessing_file.py 

import sys, csv, codecs, re
from math import log
from sklearn.externals import joblib
import pickle
import unicodedata

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def remove_accents(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

dir1 = "../data_sachin/"
trainFile = ''.join([dir1,'AZ_train_data_till_aug13.csv'])
stopFile = ''.join([dir1,'stopwords_upd.csv'])
mapFile = ''.join([dir1,'word_mapping_KW_upd.csv'])
outFile1 = ''.join([dir1,'train_data_az.csv'])

punctuation = re.compile(r'[-.?!,":;()|<>0-9*\\/~`+=]')

stopwords = csv.reader(open(stopFile,"rb")).next()
#print stopwords


####### Reading Mapping file ###########
print "Reading word map file"
map_file = open(mapFile)
map_word ={}
for line in map_file:
    k, v = line.strip().split(',')
    map_word[k.strip().lower()] = v.strip().lower()
map_file.close()


outWriter_fp1 = open(outFile1, 'wb')
outWriter1 = csv.writer(outWriter_fp1)

process_line = ["CASE_ID_","SUMMARY","PRG"]
outWriter1.writerow(process_line)
### Go to each word and do preprocessing
print "Doing preprocessing...",
sys.stdout.flush()
counter = 1
for label in [1]:
    dataFile = csv.reader(open(trainFile, 'rb'))

    header = dataFile.next()
    fields = dict(zip(header,range(len(header))))
    
    for item in dataFile:
        try:
            text = item[fields['SUMMARY']].decode('utf-8')
        except UnicodeDecodeError:
            print "UnicodeDecodeError on entry:",item[fields['CASE_ID_']]
            #sys.exit(0)
        process_line = []
        process_line.append(item[fields['CASE_ID_']].decode('utf-8'))


        # to lower case, remove punctuation
        text = text.lower()
        text = punctuation.sub(" ",text)
        text = text.replace("'s", ' ')
        text = text.replace("'", ' ')
        text = strip_accents(text)
        text = remove_accents(text)
        text1 = ""
        for word1 in text.split(" "):
            word1 = word1.strip()
            if word1 in map_word.keys():
                word = map_word[word1]
            else:
                word = word1
            # remove "common" words, aka stopwords
            if word in stopwords or word==" ": continue
        if len(word) == 1: continue
        text1 = text1 + " " + word.encode('utf-8')
        process_line.append(text1)
    process_line.append(item[fields['PRG']])
    #process_line.append(item[fields['CATEGORIZATION_TIER_1']])
    #process_line.append(item[fields['CATEGORIZATION_TIER_2']])
    #process_line.append(item[fields['CATEGORIZATION_TIER_3']])
    outWriter1.writerow(process_line)
    if counter%1000 ==0:
        print counter
        counter = counter+1
        
outWriter_fp1.close()
