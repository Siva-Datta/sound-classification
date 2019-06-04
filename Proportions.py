import matplotlib.pyplot as plt
import librosa 
import os
import numpy as np
import pandas as pd
import timeit
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  MinMaxScaler
import heapq
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import datasets, model_selection
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def prop(types,y,lbl):
	per=[]
	for proportion in types:
		score=0	
		for i in y:
			if(proportion == i):
				score=score+1	
		per.append(score)
	per=np.array(per)/sum(np.array(per))
	per=np.around(per,decimals=3)	
	table=dict(zip(types,per))
	return print('\n\nProportions of each Class in '+lbl+ ' Data are :\n{}'.format(table))





data=pd.read_csv('train/train.csv')

print(data)

y=list(data['Class'].values)


types=list(data.Class.unique())
prop(types,y,'total')
y_tr,y_ts = train_test_split(y,test_size=0.25,random_state=22)
prop(types,y_tr,'Train')
prop(types,y_ts,'Test')

