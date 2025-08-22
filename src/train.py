#!/usr/bin/env python
# coding: utf-8

# In[3]:





# In[1]:


import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score ,roc_auc_score 
from sklearn.linear_model import LogisticRegression, LinearRegression  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
import matplotlib.pyplot as plt 
import seaborn as sns  
from sklearn.naive_bayes import MultinomialNB 


# In[2]:


data = pd.read_csv("heart_disease_uci.csv") 

print(data.head())  


data['target'] = data['num']  


if 'id' in data.columns:
    data = data.drop(columns=['id']) 

numeric_columns = data.select_dtypes(include=['number']).columns

data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)  

data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())  


categorical_columns = ['sex', 'dataset', 'cp', 'restecg', 'exang', 'slope', 'thal'] 
label_encoders = {}  
for column in categorical_columns:
    encoder = LabelEncoder()  
    data[column] = encoder.fit_transform(data[column]) 
    label_encoders[column] = encoder 
    
# Separate features (X) and target (y)
X = data.drop(columns=['num', 'target']) 
y = data['target']  


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)


scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test) 

if np.isnan(X_train).any() or np.isinf(X_train).any():
    print("Warning: X_train contains NaN or infinity values. Replacing them with zeros.")
    X_train = np.nan_to_num(X_train)  # Replace NaN and infinity with zeros

if np.isnan(X_test).any() or np.isinf(X_test).any():
    print("Warning: X_test contains NaN or infinity values. Replacing them with zeros.")
    X_test = np.nan_to_num(X_test)  # Replace NaN and infinity with zeros

# Confirm preprocessing
print("Preprocessing complete.")


# In[8]:


data.info()


# In[3]:


# Initialize the Random Forest model
# grid search
rf_model = RandomForestClassifier(random_state=42)  

rf_model.fit(X_train, y_train) 


rf_score = rf_model.score(X_test, y_test) 

rf_predictions = rf_model.predict(X_test)  

# Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_predictions) 
rf_error = 1 - rf_accuracy  
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)  
rf_precision = precision_score(y_test, rf_predictions, average='weighted', zero_division=0)  
rf_recall = recall_score(y_test, rf_predictions, average='weighted', zero_division=0)  
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')  
rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')  

# Display metrics
print("Random Forest Results:")
print(f"Score: {rf_score:.4f}") 
print(f"Accuracy: {rf_accuracy:.4f}") 
print(f"Error: {rf_error:.4f}") 
print(f"Precision: {rf_precision:.4f}") 
print(f"Recall: {rf_recall:.4f}")
print(f"F1 Score: {rf_f1:.4f}") 
print(f"ROC AUC: {rf_roc_auc:.4f}") 
print("Confusion Matrix:")  
print(rf_conf_matrix) 


# In[4]:


# Initialize the SVM model
svm_model = SVC(probability=True, random_state=42)  

svm_model.fit(X_train, y_train)  

svm_score = svm_model.score(X_test, y_test)  

svm_predictions = svm_model.predict(X_test)

# Calculate metrics
svm_accuracy = accuracy_score(y_test, svm_predictions) 
svm_error = 1 - svm_accuracy 
svm_conf_matrix = confusion_matrix(y_test, svm_predictions) 
svm_precision = precision_score(y_test, svm_predictions, average='weighted', zero_division=0)  
svm_recall = recall_score(y_test, svm_predictions, average='weighted', zero_division=0) 
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')  
svm_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test), multi_class='ovr')  

# Display metrics
print("SVM Results:")
print(f"Score: {svm_score:.4f}")  
print(f"Accuracy: {svm_accuracy:.4f}") 
print(f"Error: {svm_error:.4f}") 
print(f"Precision: {svm_precision:.4f}")  
print(f"Recall: {svm_recall:.4f}")  
print(f"F1 Score: {svm_f1:.4f}")  
print(f"ROC AUC: {svm_roc_auc:.4f}") 
print("Confusion Matrix:") 
print(svm_conf_matrix) 


# In[ ]:





# In[ ]:





# In[ ]:




