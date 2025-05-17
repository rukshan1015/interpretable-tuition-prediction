#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing main libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # for saving/loading the model

#MImporting libries for feature engineering and Model training

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score
import xgboost as xgb

# PDP and SHAP

from sklearn.inspection import PartialDependenceDisplay
import shap



# In[2]:


# Importing Data

data = pd.read_csv(r"C:\Users\ruksh\Desktop\Data_science\Data\International_Education_Costs.csv")


# In[3]:


## EDA - Statistics

#Printing column names
print(data.columns)

#Data types of each column
print(data.dtypes)

#Check for missing values
print(data.isna().sum())

#Data statistics
print(data.describe())

print(data.nunique)


# In[4]:


# Column selection and categorization 
x=data.drop('Tuition_USD',axis=1)
y=data['Tuition_USD']

cat=['Country', 'City', 'University', 'Program', 'Level']
num=['Duration_Years','Living_Cost_Index', 'Rent_USD', 'Visa_Fee_USD','Insurance_USD', 'Exchange_Rate']



# In[5]:


# EDA - Plots (Tuition variation with categorical variables - barplots)

for category in cat: 
    top_10 = pd.DataFrame(
    data.groupby(category)['Tuition_USD'].mean().nlargest(10)
    ).reset_index()

    sns.barplot(x=category,y='Tuition_USD',data=top_10)
    plt.xticks(rotation=90)
    plt.title(f'{category} with Highest Average Tuition')

    #plt.tight_layout()
    plt.show()


# In[6]:


## EDA - Plots (Pairwise relationships/correlations between numberical variables)

data_numerical = data[['Duration_Years','Tuition_USD', 'Living_Cost_Index', 'Rent_USD', 'Visa_Fee_USD','Insurance_USD', 'Exchange_Rate']]
sns.heatmap(data_numerical.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[7]:


# EDA - Plots (Pairplots to observe relationships)

sns.pairplot(data_numerical)


# In[8]:


## Determining levels in each categorical column. 

# This is important in selecting the proper encoding method. For example, large number of unique levels leads to high-cardinality categoricals.

print(data[cat].nunique())


# In[9]:


# Given the high cardinality in categorical data, frequescy ecoding is more suitable for intepretability 

# Creating a custom encoder for frequency encoding 
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)  # Ensure DataFrame
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps[col] = freq
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.freq_maps:
                X[col] = X[col].map(self.freq_maps[col]).fillna(0)
            else:
                X[col] = 0  # or keep original if preferred
        return X.values  # Return numpy array for sklearn compatibility

    def get_feature_names_out(self, input_features=None):
        return input_features

# Feature engineering 
ct = ColumnTransformer(transformers=[
                       ('num',StandardScaler(),num),
                       ('cat',FrequencyEncoder(),cat),
                        ])

# Model/pipeline training

model_pipeline=Pipeline([
    ('preprocessing',ct),
    ('model',xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1))
])



# In[10]:


# Splitting the datasetand Training 
X_train, X_test, y_train, y_test = split(x, y, test_size=0.2, random_state=42)



# In[11]:


# Transforming and Training in pipeline

model_pipeline.fit(X_train,y_train)


# In[12]:


# Evaluation

predicted = model_pipeline.predict(X_test)

r2_score(y_test, predicted)


# In[13]:


# PDP - If all else is kept constant, how does changing one feature affect the model's prediction?

X_test_transformed = model_pipeline.named_steps['preprocessing'].transform(X_test)
xgb_model = model_pipeline.named_steps['model']
feature_names = model_pipeline.named_steps['preprocessing'].get_feature_names_out()
features = list(range(len(feature_names)))

# Use transformed X data
PartialDependenceDisplay.from_estimator(
    estimator=xgb_model,
    X=X_test_transformed,                   
    features=features,
    feature_names=feature_names,               
    grid_resolution=50,
)

plt.subplots_adjust(hspace=0.8, wspace=0.8)
plt.show()
plt.savefig("outputs/PDP_plot.png")

## IMPORTANT

# PDP shows the average effect of a feature on predictions, assuming others stay constant.
# May be misleading when features are highly correlated (e.g., Living_Cost_Index, Rent, Insurance).


# In[ ]:


## Model explainability with SHAP (how much each feature contributes to increasing or decreasing a prediction — for each individual data point.) 

# Create SHAP explainer
explainer = shap.Explainer(xgb_model, X_test_transformed)

# Compute SHAP values
shap_values = explainer(X_test_transformed)

shap_values.feature_names = feature_names

# Plot global interpretation
shap.plots.beeswarm(shap_values)
plt.savefig("outputs/shap_plot.png")

# === Summary ===
# - Feature importance (SHAP): 'cat__Country' has the highest contribution to predictions.
# - PDP showed limited effect for 'Living_Cost_Index' due to correlation with Rent and Insurance.
# - SHAP provided clearer, more accurate interpretability than XGBoost's built-in feature importance.


# In[ ]:


#Saving the model

joblib.dump(model_pipeline, "outputs/model_pipeline.joblib")


# In[ ]:


#Loading the model and inferencing with the loaded model with new data

#load_pipeline = joblib.load("model_pipeline.joblib")
#new_prediction = load_pipeline.predict(X_test)




