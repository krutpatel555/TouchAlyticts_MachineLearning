# Import required libraries 
import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
ta = pd.read_csv("C:\Users\Krut Patel\Downloads\FeatureExt.csv") 
touchdata = pd.read_csv("C:\Users\Krut Patel\Downloads\FeatureExt.csv")

# Assign column names to the dataframe
touchdata.columns = ["User Id","Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]
ta.columns = ["User Id","Doc ID", "Inter-stroke time","Stroke Duration","START X", "Start y","Stop x", "Stop y",
            "Direct End-to-End Distance","Mean Resultant Length"," Up/Down/Left/Right Flag", "DIrection of End-to-End Line",
            "Phone ID","20 perc. Pairwise Velocity","50 perc. Pairwise Velocity","80 perc. Pairwise Velocity",
            "20 perc. pairwise acc","50 perc. pairwise acc","80 perc. pairwise acc", "median velocity at last 3 pts",
            "largest deviation from end-to-end line","20 perc. dev. from end-to-end line","50 perc. dev. from end-to-end line",
            "80 perc. dev. from end-to-end line","average direction","length of trajectory","ratio end-to-end dist and length of trajectory",
            "average velocity","median acceleration at first 5 points","mid-stroke pressure","mid-stroke area covered",
            "mid-stroke finger orientation","change of finger orientation","phone orientation"]

# Inspect first few rows of dataset
touchdata.head()

# Get number of rows and columns
touchdata.shape

# Replace infinity values with NaN  
ta.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with mean  
ta= ta.fillna(ta.mean())

# Check for null values
ta.isnull().sum()

# Generate statistics summary  
touchdata.describe()

# Drop rows with NaN values
touchdata.dropna(inplace=True)

# Extract label column  
UserID = touchdata['User Id']  
UserID_2 = ta['User Id']

# Drop label column from datasets 
touchdata.drop('User Id', axis=1, inplace=True) 
ta.drop('User Id', axis=1, inplace=True)

# Check for null values  
touchdata.isnull().sum()

# Visualize relationship between features
sns.boxplot(x='User Id', y='Up/Down/Left/Right Flag', data=touchdata) 
sns.barplot(x='User Id', y='mid-stroke pressure', data=touchdata)
sns.boxplot(y='Direct End-to-End Distance', x='User Id', data=touchdata)
sns.lineplot(y="20 perc. Pairwise Velocity", x='Stop x', data=touchdata) 

# View feature correlations  
sns.heatmap(touchdata.corr(), annot=True, cmap='coolwarm')


# Scale data for processing  
touchdata_mm = MinMaxScaler().fit_transform(touchdata)  
ta_mm = MinMaxScaler().fit_transform(ta) 
touchdata_ss = StandardScaler().fit_transform(touchdata)
ta_ss = StandardScaler().fit_transform(ta)

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Define dictionary of models
models = {
  'LogisticRegression': LogisticRegression(),
  'KNeighborsClassifier': KNeighborsClassifier(),
  'GaussianNB': GaussianNB(),
  'SVC': SVC(),
  'DecisionTreeClassifier': DecisionTreeClassifier(), 
  'RandomForestClassifier': RandomForestClassifier(),
  'MLPClassifier': MLPClassifier()
}

# Function to calculate model scores
def calculate_scores(ta, label, models):
  
  #Split data
  
  X_train, X_test, y_train, y_test = train_test_split(ta, label, test_size=0.3, random_state=42) 
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  

  # SVM Grid Search
svm_params = {'C': [0.01, 0.1, 10, 100], 
              'gamma': [10, 1, 0.1, 0.01],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

svm_gs = GridSearchCV(estimator=SVC(), param_grid=svm_params, cv=5)
svm_gs.fit(X_train, y_train) 
best_svm = svm_gs.best_estimator_

# Evaluate SVM 
svm_accuracy = evaluate(best_svm, X_val, y_val, X_test, y_test)

# KNN Grid Search
knn_params = {'n_neighbors': [10, 11],
              'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree', 'brute']} 

knn_gs = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_params, cv=5)
knn_gs.fit(X_train, y_train)
best_knn = knn_gs.best_estimator_

# Evaluate KNN
knn_accuracy = evaluate(best_knn, X_val, y_val, X_test, y_test)

# Evaluation function
def evaluate(model, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)  
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    return [val_accuracy, test_accuracy]

# MLP Grid Search
mlp_params = {'hidden_layer_sizes':[(128, 128, 128)],  
              'activation':['relu'],
              'solver':['adam'], 
              'max_iter':[100, 500, 1000, 2000],
              'learning_rate':['constant','adaptive'],}  

mlp_gs = GridSearchCV(MLPClassifier(), param_grid=mlp_params, cv=5)
mlp_gs.fit(X_train, y_train)

best_mlp = mlp_gs.best_estimator_

# Evaluate MLP
mlp_accuracy = evaluate_model(best_mlp, X_val, y_val, X_test, y_test)

# Evaluation function
def evaluate_model(model, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    return [val_accuracy, test_accuracy]
