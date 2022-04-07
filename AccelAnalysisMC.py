# import dataset + libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
%matplotlib inline
df = pd.read_csv('AccelData.csv')

print(df.info())

df=df.truncate(before=0, after=109)
df.info()

maxRows=len(df)
time=((1/6)*np.arange(maxRows))
df['time']=time
df.head()

for column in df:
    df.plot(kind='line',x='time',y=[column])

df = df.iloc[20:]
for column in df:
    df.plot(kind='line',x='time',y=column)

del df['time']
df.head()

averageStat=[0]*12
stDevStat=[0]*12
diffStat=[0]*12
columnN=0
for column in df:
    count2=0
    tempList=[0] * 10
    average=[0] * 11
    stDev=[0] * 11
    diff=[0]*11 #max-min
    count=0
    for n in df[column]:
        tempList[count]=n
        if(count!=0 and (count+1)%10==0):
            average[count2]=sum(tempList)/len(tempList)
            stDev[count2]=np.std(tempList)
            diff[count2]=max(tempList)-min(tempList)
            count=-1
            count2+=1
        count+=1
    averageStat[columnN]=average
    stDevStat[columnN]=stDev
    diffStat[columnN]=diff
    print(columnN)
    columnN+=1
print(averageStat)
print(stDevStat)
print(diffStat)

columnNames=[None]*12
count=0
for column in df:
    columnNames[count]=column
    count+=1
averageAverage=[0]*len(averageStat)
averageStDev=[0]*len(averageStat)
averageRange=[0]*len(averageStat)
stDevAverage=[0]*len(averageStat)
stDevStDev=[0]*len(averageStat)
stDevRange=[0]*len(averageStat)
count=0
while count<len(averageStat):
    averageAverage[count]=sum(averageStat[count])/len(averageStat[count])
    averageStDev[count]=sum(stDevStat[count])/len(stDevStat[count])
    averageRange[count]=sum(diffStat[count])/len(diffStat[count])
    stDevAverage[count]=np.std(averageStat[count])
    stDevStDev[count]=np.std(stDevStat[count])
    stDevRange[count]=np.std(diffStat[count])
    count+=1

fig, ax = plt.subplots()
ax.bar(np.arange(len(averageAverage)),averageAverage,yerr=stDevAverage, align='center')
ax.set_ylabel("Mean averages of windows")
ax.set_xticks(np.arange(len(averageAverage)))
ax.set_xticklabels(columnNames)
ax.yaxis.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.bar(np.arange(len(averageAverage)),averageStDev,yerr=stDevStDev, align='center')
ax.set_ylabel("Mean standard deviation")
ax.set_xticks(np.arange(len(averageAverage)))
ax.set_xticklabels(columnNames)
ax.yaxis.grid(True)
plt.show()

fig, ax = plt.subplots()
ax.bar(np.arange(len(averageAverage)),averageRange,yerr=stDevRange, align='center')
ax.set_ylabel("Mean range")
ax.set_xticks(np.arange(len(averageAverage)))
ax.set_xticklabels(columnNames)
ax.yaxis.grid(True)
plt.show()

classes=[0]*22+[1]*44+[0]*66
averages=[0]*132
stDevs=[0]*132
ranges=[0]*132
count=0
count3=0
while count<12:
    count2=0
    while count2<11:
        print(count3)
        averages[count3]=averageStat[count][count2]
        stDevs[count3]=stDevStat[count][count2]
        ranges[count3]=diffStat[count][count2]
        count2+=1
        count3+=1
    count+=1
print(averages)

df2=pd.DataFrame(list(zip(averages, stDevs, ranges)),
              columns=['Average','Standard Deviation', 'Range'])
print(df2)
x=df2
y=classes
yMultiple = ["Writing"]*22+["Clapping"]*22+["Hand Flapping"]*22+["Walking"]*22+["Jumping Jacks"]*22+["Descending Stairs"]*22

print(yMultiple)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
XM_train, XM_test, yM_train, yM_test = train_test_split(x, yMultiple, test_size = 0.20)
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
svclassifierM = SVC(kernel='poly', degree=20)
svclassifierM.fit(XM_train, yM_train)

y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

yM_pred = svclassifierM.predict(XM_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(yM_test,yM_pred))
print(classification_report(yM_test,yM_pred))

from sklearn.externals import joblib
joblib.dump(svclassifier, 'firstModel.pkl')

xTest=[[1.064279193,
0.338470518,
0.083432]]
y=svclassifierM.predict(xTest)
print(y)

xTest=[[1.5,
0.5,
1.5]]
y=svclassifierM.predict(xTest)
print(y)

#just testing if it predicts 2 classes
import random
for x in range(50):
    xTest=[[random.uniform(1.0,2.5),random.uniform(0.05,0.8),random.uniform(0.1,3.0)]]
    print(xTest,svclassifier.predict(xTest))

#just testing if it predicts 2 classes
import random
for x in range(50):
    xTest=[[random.uniform(1.0,2.5),random.uniform(0.05,0.8),random.uniform(0.1,3.0)]]
    print(xTest,svclassifierM.predict(xTest))
