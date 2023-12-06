

#importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#reading the dataset
df= pd.read_csv("C:\\Users\\jinal\\Downloads\\ML Project\\FeatMat.csv",header=None)
dataset = pd.read_csv("C:\\Users\\jinal\\Downloads\\ML Project\\FeatMat.csv",header=None)

#assigning the features name
dataset.columns = ["User Id","Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]
df.columns = ["User Id","Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]

#Visualizing the few rows of dataset
dataset.head()

#Getting total number of rows and columns of dataset(version 1)
dataset.shape

#replacing the infinity values in version 2 with NaN
import numpy as np
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#replacing the null values with mean of the column
df= df.fillna(df.mean())

df.shape

#checking for null values in version 2
df.isnull().sum()

#getting data statistics of version1 of dataset
dataset.describe()

#checking for null values in version 1
dataset.isnull().sum()

dataset.dropna(inplace=True) #dropping all the null values in version 1 of dataset
UID = dataset['User Id'] #creating label of version 1
UID_2= df['User Id'] #creating label of version 2
#droppingthe label
dataset.drop('User Id',axis=1,inplace=True)
df.drop('User Id',axis=1,inplace=True)

dataset.isnull().sum()

sns.boxplot(x='User Id',y=' Up/Down/Left/Right Flag',data=dataset)



sns.barplot(x='User Id',y='mid-stroke pressure',data=dataset)



sns.boxplot(y='Direct End-to-End Distance',x='User Id',data=dataset)


sns.lineplot(y="20 perc. Pairwise Velocity",x='Stop x',data=dataset)


sns.pairplot(dataset)

sns.set(rc={"figure.figsize":(26, 25)}) #width=6, height=5

dataplot = sns.heatmap(dataset.corr(),annot=True,cmap='coolwarm')



#sScaling the dataset using MinMax scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

dataset_mm= MinMaxScaler().fit_transform(dataset)
df_mm=MinMaxScaler().fit_transform(df)

dataset_mm= pd.DataFrame(dataset_mm)
df_mm= pd.DataFrame(df_mm)


dataset_mm.columns = ["Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]
df_mm.columns = ["Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]

#Scaling the data using standardScaler
dataset_ss= StandardScaler().fit_transform(dataset)
df_ss= StandardScaler().fit_transform(df)



dataset_ss= pd.DataFrame(dataset_ss)
df_ss= pd.DataFrame(df_ss)



dataset_ss.columns = ["Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]
df_ss.columns = ["Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]

#importing the classifiers
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

from sklearn.naive_bayes import GaussianNB
gnb= GaussianNB()

from sklearn.svm import SVC
svm=SVC()

from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier()

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()
mlp.out_activation_ = 'softmax'

# Making dictonary of models
models={'lr':lr,
       'KNN':knn,
       'GNB':gnb,
       'SVM':svm,
       'DTC':dtc,
       'RFC':rfc,
       'MLP':mlp
       }
models

#creating function to calculate the score
def function(df,label,ta):
    from sklearn.model_selection import train_test_split
    import sklearn
    train_x,test_x,train_y,test_y =  train_test_split(df,label,test_size=0.3, random_state=42)
    train_X,val_X,train_Y,val_Y =  train_test_split(train_x,train_y,test_size=0.2, random_state=42)
    training={}



    for J in ta:
        acc=[]
        ta[J].fit(train_X,train_Y)
        p_train=ta[J].predict(train_X)
        #ta[J].fit(val_X,val_Y)
        p_val=ta[J].predict(val_X)
        #ta[J].fit(test_x,test_y)
        p_test=ta[J].predict(test_x)
        #training
        acc.append(sklearn.metrics.accuracy_score(train_Y,p_train))
        #print(sklearn.metrics.classification_report(train_Y,p_train))
        #validation
        acc.append(sklearn.metrics.accuracy_score(val_Y,p_val))
        #val.append(sklearn.metrics.classification_report(val_Y,p_val))
        #testing
        acc.append(sklearn.metrics.accuracy_score(test_y,p_test))
        #test.append(sklearn.metrics.classification_report(test_y,p_test))

        training[J]=acc

    training= pd.DataFrame(training, index=['train','val','test'])

    return training

#calculating the score for version 1
function(dataset_ss,UID,models)


#calculating scores for Version 2
function(df_mm,UID_2,models)


#Splitting the data for hyperparameter tuning
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y =  train_test_split(dataset_ss,UID,test_size=0.3, random_state=42)
train_X,val_X,train_Y,val_Y =  train_test_split(train_x,train_y,test_size=0.2, random_state=42)

t_x,te_x,t_y,te_y= train_test_split(df_mm,UID_2,test_size=0.3,random_state=42)
t_X,v_x,t_Y,v_y= train_test_split(t_x,t_y,test_size=0.2,random_state=42)
results={'SVM':[],'KNN':[],'MLP':[]} #making empty dictonary

#SVM Hyperparameter tuning
from sklearn.model_selection import (GridSearchCV, StratifiedKFold)
svm_grid = {'C': [0.01,0.1, 10,100],
              'gamma': [10,1, 0.1,0.01],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
svm_cv = GridSearchCV(estimator=SVC(),
                     param_grid=svm_grid,
                     refit='auc',
                     cv = StratifiedKFold(n_splits=5, random_state=42,shuffle=True))
svm_cv.fit(train_X,train_Y)

svm_cv.best_params_

svm= SVC(C=svm_cv.best_params_['C'], gamma=svm_cv.best_params_['gamma'], kernel= svm_cv.best_params_['kernel'])

hp_tune=[]
import sklearn
svm.fit(train_X,train_Y)
predict_val=svm.predict(val_X)
predict_test= svm.predict(test_x)

hp_tune.append(sklearn.metrics.accuracy_score(val_Y,predict_val))
hp_tune.append(sklearn.metrics.accuracy_score(test_y,predict_test))
results['SVM']=hp_tune

#KNN hyperparameter tuning
knn_grid = {'algorithm': ['ball_tree','kd_tree','brute'],
            'leaf_size':[30],
            'metric':['minkowski'],
            'weights': ['uniform','distance'],
            'n_neighbors': [10,11]
           }
knn_cv = GridSearchCV(estimator=KNeighborsClassifier(),
                     param_grid=knn_grid,
                     refit='auc',
                     cv = StratifiedKFold(n_splits=5, random_state=42,shuffle=True))
knn_cv.fit(t_X,t_Y)

knn_cv.best_params_

knn= KNeighborsClassifier(algorithm=knn_cv.best_params_['algorithm'],leaf_size=knn_cv.best_params_['leaf_size'],
                          metric=knn_cv.best_params_['metric'],n_neighbors=knn_cv.best_params_['n_neighbors'],
                         weights=knn_cv.best_params_['weights'])

#knn=KNeighborsClassifier()
hp_tune=[]
knn.fit(t_X,t_Y)
predict_train= knn.predict(t_X)
predict_val=knn.predict(v_x)
predict_test= knn.predict(te_x)
#print(sklearn.metrics.accuracy_score(t_Y,predict_train))
hp_tune.append(sklearn.metrics.accuracy_score(v_y,predict_val))
hp_tune.append(sklearn.metrics.accuracy_score(te_y,predict_test))
results['KNN']=hp_tune

#MLP after increasing the maximum iteration
hp_tune=[]
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(max_iter=2000,learning_rate='adaptive',early_stopping=True)
mlp.out_activation_ = 'softmax'
mlp.fit(train_X,train_Y)

predict_val=mlp.predict(val_X)
predict_train=mlp.predict(test_x)

hp_tune.append(sklearn.metrics.accuracy_score(val_Y,predict_val))
hp_tune.append(sklearn.metrics.accuracy_score(test_y,predict_train))
results['MLP']=hp_tune

f_results=pd.DataFrame(results,index=['Validation','Testing'])

f_results
