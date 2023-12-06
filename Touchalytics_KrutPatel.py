# Import required libraries 
import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
ta = pd.read_csv("C:\\Users\\Krut Patel\\Downloads\\\\FeatureExt.csv") 
touchdata = pd.read_csv("C:\\Users\\Krut Patel\\Downloads\\\\FeatureExt.csv")

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
  
  #Calculate scores
  scores = {}
  
  for name, model in models.items():
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)          
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred) 
    test_accuracy = accuracy_score(y_test, test_pred)
    
    scores[name] = [train_accuracy, val_accuracy, test_accuracy]
  
  scores = pd.DataFrame(scores, index=['train', 'val', 'test'])
  
  return scores

# Evaluate models on datasets  
scores_v1 = calculate_scores(touchdata_ss, UserID, models) 
scores_v2 = calculate_scores(ta_mm, UserID_2, models)

# Hyperparameter tuning for SVM and KNN 
tuned_models = {
  'SVM': GridSearchCV(SVR()).fit(X, y).best_estimator_,
  'KNN': GridSearchCV(KNeighborsClassifier()).fit(X, y).best_estimator_  
}

# Evaluate tuned models
final_scores = calculate_scores(ta, UserID, tuned_models)

final_scores
