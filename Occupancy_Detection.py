# -*- coding: utf-8 -*-
#import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sb
import time
import matplotlib.pyplot as plt  

##########################################00#########################################

Algorithms=['Decision Tree','ANN','Logistic Regression']
training_time =[]
test1_time=[]
test2_time=[]
test1_accur=[]
test2_accur=[]
##Read Train Dataset###
ds_train = pd.read_csv('datatraining10.csv' ,index_col ='date' ,parse_dates = True)
###Split train Data into X,and Label y
X_train = ds_train.iloc[:,2:6]
y_train=ds_train.iloc[:,6]

##Read Test Dataset##
ds_test1 = pd.read_csv('datatest10.csv',index_col ='date' ,parse_dates = True )
X_test1 = ds_test1.iloc[:,2:6]
y_test1=ds_test1.iloc[:,6]
###Delete unconcerned colomn
del ds_train['Unnamed: 0']
del ds_test1['Unnamed: 0']
##count Y labels
ds_train['Occupancy'].value_counts()

sb.countplot(x=ds_train['Occupancy'],data = ds_train,palette='hls')
plt.show();
plt.savefig('count_plot')
##show features effect (mean) on occupancy
ds_train.groupby('Occupancy').mean()



###Visualization #######




ds_train['Temperature'].plot()
ds_train['Humidity'].plot()
ds_train['Light'].plot()
ds_train['CO2'].plot()
ds_train['HumidityRatio'].plot()
ds_train['Occupancy'].plot()
###Feature Engineering###
###Count null value(indesirable values)
ds_train.isnull().sum() #compter le nombre nan
ds_test1.isnull().sum()

#####Scaling training & Test Data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test1 = sc.transform(X_test1)

pd.crosstab( ds_train['Temperature'],ds_train['Occupancy']).plot()
pd.crosstab( ds_train['CO2'],ds_train['Occupancy']).plot()
pd.crosstab( ds_train['Light'],ds_train['Occupancy']).plot()
pd.crosstab( ds_train['Humidity'],ds_train['Occupancy']).plot()
pd.crosstab( ds_train['HumidityRatio'],ds_train['Occupancy']).plot()
plt.title ('Occupancy Frequence') 
plt.savefig('Ocuupancy_Frequence')


####let's start applying some Algo###
############################################01#########################################
print('Decision Tree')
c1 =time.clock()
cls = DecisionTreeClassifier()
cls = cls.fit(X_train,y_train)
c2 =time.clock()
print('Learning time: ',c2-c1)
training_time.append(c2-c1)

 
c1 =time.clock()
res1 =cls.predict( X_test1)
scores1 = accuracy_score(y_test1, res1)
c2 =time.clock()
print('Prediction time for the first set: ',c2-c1)
print('Prediction accuracy First set: ',scores1) 
test1_time.append(c2-c1)
test1_accur.append(scores1) 
##calssification report#### 
from sklearn.metrics import classification_report 
print (classification_report (y_test1, res1))



# Making the Confusion Matrix  Metrics --> quality of prediction
from sklearn.metrics import confusion_matrix 
cm1 = confusion_matrix(y_test1, res1)
print(cm1)






##################################### ANN  ##############################################"
print('ANN')

c1 =time.clock() 
cls = MLPClassifier(hidden_layer_sizes=(300,300,300,300),activation ='relu',solver='sgd',batch_size= 200,learning_rate='adaptive',shuffle=True,verbose=False)
cls = cls.fit(X_train,y_train)
c2 =time.clock()
print('Learning time: ',c2-c1)
training_time .append(c2-c1)
 
c1 =time.clock() 
res3 =cls.predict( X_test1)
scores1 = accuracy_score(y_test1, res3)
c2 =time.clock()
print('Prediction time for the first set: ',c2-c1)
print('Prediction accuracy First set: ',scores1) 
test1_time.append(c2-c1)
test1_accur.append(scores1) 

# Making the Confusion Matrix  Metrics --> quality of prediction
from sklearn.metrics import confusion_matrix 
cm2 = confusion_matrix(y_test1, res3)
print(cm2)

##################################### Logistic Regression  ##############################################"
print('Logistic Regression')

c1 =time.clock() 
classifier = LogisticRegression(solver ='sag' )
classifier.fit(X_train, y_train)
c2 =time.clock()
print('Learning time: ',c2-c1)
training_time .append(c2-c1)
 
c1 =time.clock() 
# Predicting the Test set results
res4 = classifier.predict(X_test1)
scores1 = accuracy_score(y_test1, res4)
scores1
c2 =time.clock()
print('Prediction time for the first set: ',c2-c1)
print('Prediction accuracy First set: ',scores1) 
test1_time.append(c2-c1)
test1_accur.append(scores1) 

##wights
print(classifier.coef_)

# Making the Confusion Matrix  Metrics --> quality of prediction
from sklearn.metrics import confusion_matrix 
cm3 = confusion_matrix(y_test1, res4)
print(cm3)

#####################################Viz#####################################
  
plt.bar(range(len(training_time)), training_time, align='center')
plt.xticks(range(len(training_time)), Algorithms, size='small')
plt.title('Training Time')
plt.show()


plt.bar(range(len(test1_time)), test1_time, align='center')
plt.xticks(range(len(test1_time)), Algorithms, size='small')
plt.title('Test Time')
plt.show()


plt.bar(range(len(test1_accur)), test1_accur, align='center')
plt.xticks(range(len(test1_accur)), Algorithms, size='small')
plt.title('test Accuracy')
plt.show()


