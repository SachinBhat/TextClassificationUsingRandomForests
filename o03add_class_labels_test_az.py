# preprocessing_file.py 

import sys, csv, codecs, re
from math import log
from sklearn.externals import joblib
import pickle

dir1 = "../data_sachin/"
posFile = ''.join([dir1,'test_data_az_v2.csv'])
prgFile = ''.join([dir1,'PRG_mapping_az.csv'])
ct1File = ''.join([dir1,'cat_tier1_mapping_az.csv'])
ct2File = ''.join([dir1,'cat_tier2_mapping_az.csv'])
ct3File = ''.join([dir1,'cat_tier3_mapping_az.csv'])
outFile1 = ''.join([dir1,'test_label_vector_az_v2.csv'])
#outFile2 = ''.join([dir1,'test_cat1_features_az.csv'])

outWriter_fp1 = open(outFile1, 'wb')
outWriter1 = csv.writer(outWriter_fp1)

#outWriter_fp2 = open(outFile2, 'wb')
#outWriter2 = csv.writer(outWriter_fp2)

####### Reading PRG Mapping file  ###########
print "Reading PRG map file"
map_file = open(prgFile)
map_word ={}
process_line1 = ["Other PRGs"]
for line in map_file:
    k, v = line.strip().split(',')
    map_word[k] = v
    process_line1.append(k)
map_file.close()

####### Reading Category Tier1 Mapping file  ###########
##print "Reading Category Tier1 mapping file"
##ct1map_file = open(ct1File)
##ct1map_word ={}
##process_line2 = ["Catg T1 Others"]
##for line in ct1map_file:
##    k, v = line.strip().split(',')
##    ct1map_word[k] = v
##    process_line2.append(k)
##ct1map_file.close()
##
######### Reading Category Tier2 Mapping file  ###########
##print "Reading Category Tier2 mapping file"
##ct2map_file = open(ct2File)
##ct2map_word ={}
##process_line2.append("Catg T2 Others")
##for line in ct2map_file:
##    k, v = line.strip().split(',')
##    ct2map_word[k] = v
##    process_line2.append(k)
##ct2map_file.close()
##
######### Reading Category Tier3 Mapping file  ###########
##print "Reading Category Tier3 mapping file"
##ct3map_file = open(ct3File)
##ct3map_word ={}
##process_line2.append("Catg T3 Others")
##for line in ct3map_file:
##    k, v = line.strip().split(',')
##    ct3map_word[k] = v
##    process_line2.append(k)
##ct3map_file.close()

outWriter1.writerow(process_line1)
#outWriter2.writerow(process_line2)

### Creating the PRG and Category Tier 1,2,3 label vector
print "Adding PRG and Category Tier 1,2,3 labels...",
sys.stdout.flush()
counter = 1
for label in [1]:
    if label: dataFile = csv.reader(open(posFile, 'rb'))

    header = dataFile.next()
    fields = dict(zip(header,range(len(header))))

    for item in dataFile:
        process_line1 = []
        process_line2 = []

        PRG_text = item[fields['PRG']].decode('utf-8')

        if PRG_text in map_word.keys():
            PRG_label = int(map_word[PRG_text])
        else:
            PRG_label = 0

	label_vector = [0]*(len(map_word)+1)
	label_vector[PRG_label] = 1
	for i in label_vector:
            process_line1.append(i)
        
##	ct1_text = item[fields['CATEGORIZATION_TIER_1']].decode('utf-8')
##
##        if ct1_text in ct1map_word.keys():
##            ct1_label = int(ct1map_word[ct1_text])
##        else:
##            ct1_label = 0
##
##        ct1label_vector = [0]*(len(ct1map_word)+1)
##        ct1label_vector[ct1_label] = 1
##        for i in ct1label_vector:
##            process_line2.append(i)
##
##        ct2_text = item[fields['CATEGORIZATION_TIER_2']].decode('utf-8')
##
##        if ct2_text in ct2map_word.keys():
##            ct2_label = int(ct2map_word[ct2_text])
##        else:
##            ct2_label = 0
##
##        ct2label_vector = [0]*(len(ct2map_word)+1)
##        ct2label_vector[ct2_label] = 1
##        for i in ct2label_vector:
##            process_line2.append(i)
##
##        ct2_text = item[fields['CATEGORIZATION_TIER_2']].decode('utf-8')
##
##        if ct2_text in ct2map_word.keys():
##            ct2_label = int(ct2map_word[ct2_text])
##        else:
##            ct2_label = 0
##
##        ct2label_vector = [0]*(len(ct2map_word)+1)
##        ct2label_vector[ct2_label] = 1
##        for i in ct2label_vector:
##            process_line2.append(i)


        outWriter1.writerow(process_line1)
#       outWriter2.writerow(process_line2)

        if counter%1000 ==0:
            print counter
        counter = counter+1
        
outWriter_fp1.close()
#outWriter_fp2.close()


