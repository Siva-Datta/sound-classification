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
from sklearn.metrics import roc_curve, auc

'''
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc(kernel='linear', c=1, gamma=1) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
'''
import warnings
warnings.filterwarnings("ignore")




scaler=MinMaxScaler()
tr_f=os.listdir('train/Train1/')
#tr_f=os.listdir('train/Train/')

#print(tr_f[0])

print('Number of clips used for training are:{}'.format(int(len(tr_f)*0.85)))
#tr_f.sort()



def cos_knn(k, test_data, test_target, stored_data, stored_target):
    """k: number of neighbors to use for voting
    test_data: a set of unobserved images to classify
    test_target: the labels for the test_data (for calculating accuracy)
    stored_data: the images already observed and available to the model
    stored_target: labels for stored_data
    """
#print('\nValue of k used is :',k)
    # find cosine similarity for every point in test_data between every other point in stored_data
    cosim = cosine_similarity(test_data, stored_data)
    
    # get top k indices of images in stored_data that are most similar to any given test_data point
    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    # convert indices to numbers using stored target values
    top = [[stored_target[j] for j in i[:k]] for i in top]
    
    # vote, and return prediction for every image in test_data
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)
    Report=classification_report(test_target, pred,output_dict=True)
# print table giving classifier accuracy using test_target
    return Report['accuracy']




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


fstart=timeit.default_timer()

#print('\nShowing First 5 of training set:{}'.format(tr_f[:5]))
data=pd.read_csv('train/train1.csv')
#data=pd.read_csv('train/train.csv')

print(data)

y=list(data['Class'].values)
#y=np.array(y)
#y=y.reshape(len(tr_f),1)




types=list(data.Class.unique())
print('Guessing from classes : {}'.format(types))

count=0

for file in tr_f:
	avg=[]
	x,sr=librosa.load('train/Train1/'+file)
#	x,sr=librosa.load('train/Train/'+file)

	#print(file,sr,np.shape(x))
	mfccs=librosa.feature.mfcc(x,sr=sr)
	#print(len(mfccs))
	if(count==0):
		a=[]
		for i in range(len(mfccs)):
			a.append(i)	
			b=pd.DataFrame(columns=a)
	#print(np.shape(mfccs))
	#print(np.shape(sum(mfccs)),len(sum(mfccs)))
	#print(mfccs[0],np.shape(mfccs[0]))	
	for i in range(len(mfccs)):
		avgapp=sum(mfccs[i])/len(mfccs[i])
		avg.append(avgapp)
		varapp=sum(np.power(mfccs[i]-avgapp,2))/len(mfccs[i])
		#avg.append(varapp)
#	print(mfccs,np.shape(mfccs),type(mfccs))
	avg=np.array(avg).reshape(1,len(avg))	
	#print(mfccs,np.shape(mfccs),type(mfccs))
	avg=pd.DataFrame(data=avg[0:],index=[int(file.replace('.wav',''))])	
	b=b.append(avg)
	count=count+1
	ll=int(len(tr_f)/6)	
	if(count%ll==0):
#	if(count%300==0):
		stop=timeit.default_timer()
		print('\nReached: {}'.format(count))
		print('time :{:.2f}'.format(stop-fstart))
print('Time taken for feature extraction: {:.2f}'.format(timeit.default_timer()-fstart))	
'''
file=tr_f[0]
x,sr=librosa.load('train/Train1/'+file)
mfccs=librosa.feature.mfcc(x,sr=sr)
mfccs=sum(mfccs)/len(mfccs)
mfccs=mfccs.reshape(1,len(mfccs))
mfccs=pd.DataFrame(data=mfccs[0:],index=[file.replace('.wav','')])	
b=b.append(mfccs)
count=count+1	
'''

print('\n')
b=b.sort_index(axis=0)
p=pd.DataFrame(data=y,index=list(data['ID'].values),columns=['label'])
b=b.fillna(0)
b= pd.concat([b, p], axis=1)
b[a]=scaler.fit_transform(b[a])

X=b[a]
b_tr,b_ts,y_tr,y_ts = train_test_split(X,y,test_size=0.15,random_state=22)

'''

#from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
Rstart=timeit.default_timer()
clf=RandomForestClassifier(n_estimators=100)
clf.fit(b_tr,y_tr)
score=clf.score(b_ts, y_ts)
Rstop=timeit.default_timer()
print('Time taken for Random forest : {:.2f}'.format(Rstop-Rstart),'Score of RForest: {:.2f}'.format(score))
print('Classification Report for Random Forest :- \n',classification_report(y_ts,clf.predict(b_ts)))





Rstart=timeit.default_timer()
clf=LinearSVC(random_state=0,tol=1e-5)
clf.fit(b_tr,y_tr)
Rstop=timeit.default_timer()
print('Time taken for Linear SVC : {:.2f}'.format(Rstop-Rstart),'Score of LinearSVC: {:.2f}'.format(clf.score(b_ts,y_ts)))
score=max(score,clf.score(b_ts,y_ts))
print('Classification Report for Linear SVC :- \n',classification_report(y_ts,clf.predict(b_ts)))


'''


'''
prop(types,y,'total')
prop(types,y_tr,'Train')
prop(types,y_ts,'Test')
'''



#score=max(clf.score(b_ts,y_ts),clf1.score(b_ts,y_ts))








k=int(np.power(np.array(list((b_tr).shape)),0.5)[0])
if(k%2==0):
	k=k+1

Rstart=timeit.default_timer()
scorek=0
for i in range(1,k+1):
	for j in range(2,3):

		knn = KNeighborsClassifier(n_neighbors = i,p=j)
		knn.fit(b_tr,y_tr)
		if(scorek < knn.score(b_ts,y_ts)):
			scorek = knn.score(b_ts,y_ts)
			finalk=[i,j]	
Rstop=timeit.default_timer()
print('Time taken for knn : {:.2f}'.format(Rstop-Rstart),'Score of knn: {:.2f}'.format(scorek),'[K,p] used ={}'.format(finalk))
print('Classification Report for knn :- \n',classification_report(y_ts,knn.predict(b_ts)))


'''
Rstart=timeit.default_timer()
scorec=0
for i in range(1,k+1):
	if(scorec< cos_knn(i,b_ts,y_ts,b_tr,y_tr)):
		scorec = cos_knn(i,b_ts,y_ts,b_tr,y_tr)
		kc=i
Rstop=timeit.default_timer()
print('Time taken for cnn : {:.2f}'.format(Rstop-Rstart),'Score of cnn: {:.2f}'.format(scorec),'K used ={}'.format(kc))
print('Classification Report for cnn :- \n',classification_report(y_tr,y_ts))
'''

print('Final score: {:.2f}'.format(scorek))
#print('Score is:{:.2f}'.format(score),'K used ={}'.format(finalk))		


print('Give Test ')
test=input()
print('Given input is: {}'.format(test))	

while(test!='end'):
	print('Given input is: {}'.format(test))	
	avg=[]
	x,sr=librosa.load('test/Test/'+test)
	#print(file,sr,np.shape(x))
	mfccs=librosa.feature.mfcc(x,sr=sr)
	#print(len(mfccs)
	a=[]
	for i in range(len(mfccs)):
			a.append(i)	
			nb=pd.DataFrame(columns=a)
	#print(np.shape(mfccs))
	#print(np.shape(sum(mfccs)),len(sum(mfccs)))
	#print(mfccs[0],np.shape(mfccs[0]))	
	for i in range(len(mfccs)):
		avgapp=sum(mfccs[i])/len(mfccs[i])
		avg.append(avgapp)
		varapp=sum(np.power(mfccs[i]-avgapp,2))/len(mfccs[i])
		#avg.append(varapp)
#	print(mfccs,np.shape(mfccs),type(mfccs))
	avg=np.array(avg).reshape(1,len(avg))	
	#print(mfccs,np.shape(mfccs),type(mfccs))
	avg=pd.DataFrame(data=avg[0:],index=[int(test.replace('.wav',''))])	
	nb=nb.append(avg)
	f=nb[a]
	knn = KNeighborsClassifier(n_neighbors=finalk[0],p=finalk[1])
	knn.fit(b_tr,y_tr)	
	print(knn.predict(f))
	test=input()
	knn = KNeighborsClassifier(n_neighbors=1,p=2)
	knn.fit(b_tr,y_tr)
	y=(knn.predict_proba(f)[0])

	x=list(range(1,len(types)+1))
    
	source = ColumnDataSource(data=dict(x=x, y=y))
    
    
	plot = figure(plot_width=400, plot_height=400)
	plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

	kn_slider = Slider(start=1, end=k, value=5, step=1, title="Neighbours")
	d_slider = Slider(start=1, end=5, value=2, step=1, title="Distance Metric")
	callback = CustomJS(args=dict(source=source, kn=kn_slider, d=d_slider),
                        code="""
        const data = source.data;
        const k = kn.value;
        const d = d.value;
        const x = data['x']
        const y = data['y']
        knn = KNeighborsClassifier(n_neighbors=k,p=d)
        knn.fit(b_tr,y_tr)
            for (var i = 0; i < x.length; i++) {
        y[i] = knn.predict_proba(f)[0]
    }
        
        source.change.emit();
    """)
	kn_slider.js_on_change('value', callback)
	d_slider.js_on_change('value', callback)

	layout = row(
        plot,
        column(kn_slider, d_slider),
        )

	output_file("slider.html", title="slider.py example")

	show(layout)
# The cumulative sum will be a trend line
#fig.line(x=day_num, y=cumulative_words, 
 #        color='gray', line_width=1,
  #       legend='Cumulative')

# Put the legend in the upper left corner
#fig.legend.location = 'top_left'

# Let's check it out
	print('Class with Highest Probability:{}'.format(knn.predict(f)))

'''

neighbors=range(1,k+1)
train_results = []
test_results = []
for n in neighbors:
	rf = KNeighborsClassifier(n_neighbors=n)
	rf.fit(b_tr,y_tr)	
	train_sc = rf.score(b_tr,y_tr)
	train_results.append(train_sc)
	test_sc=rf.score(b_ts,y_ts)
	test_results.append(test_sc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(neighbors, train_results, 'b', label="Train")
line2, = plt.plot(neighbors, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('score')
plt.xlabel('n')
plt.show()

ps=[1,2,3,4,5]
for pk in ps:
	rf = KNeighborsClassifier(p=pk)
	rf.fit(b_tr,y_tr)	
	train_sc = rf.score(b_tr,y_tr)
	train_results.append(train_sc)
	test_sc=rf.score(b_ts,y_ts)
	test_results.append(test_sc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(p, train_results, 'b', label="Train")
line2, = plt.plot(p, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('score')
plt.xlabel('p')
plt.show()
 


max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []

for max_depth in max_depths:
	rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
	rf.fit(b_tr, y_tr)
	train_sc = rf.score(b_tr,y_tr)
	train_results.append(train_sc)
	test_sc=rf.score(b_ts,y_ts)
	test_results.append(test_sc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label="Train")
line2, = plt.plot(max_depths, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('score')
plt.xlabel('max_depths')
plt.show()

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
	rf = RandomForestClassifier(min_samples_split=min_samples_split)
	rf.fit(b_tr, y_tr)
	train_sc = rf.score(b_tr,y_tr)
	train_results.append(train_sc)
	test_sc=rf.score(b_ts,y_ts)
	test_results.append(test_sc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('score')
plt.xlabel('min_samples_splits')
plt.show()

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
	rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
	rf.fit(b_tr, y_tr)
	train_sc = rf.score(b_tr,y_tr)
	train_results.append(train_sc)
	test_sc=rf.score(b_ts,y_ts)
	test_results.append(test_sc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(' score')
plt.xlabel('min_samples_leafs')
plt.show()

max_features = list(range(1,X.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
	rf = RandomForestClassifier(max_features=max_feature)
	rf.fit(b_tr, y_tr)
	train_sc = rf.score(b_tr,y_tr)
	train_results.append(train_sc)
	test_sc=rf.score(b_ts,y_ts)
	test_results.append(test_sc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, 'b', label="Train")
line2, = plt.plot(max_features, test_results, 'r', label="Test")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(' score')
plt.xlabel('max_features')
plt.show()

rf = RandomForestClassifier(n_estimators=32,max_depth=28,min_samples_leaf=0.2,max_features=4)
rf.fit(b_tr, y_tr)
#print(rf.score(b_ts,y_ts))
'''



