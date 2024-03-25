

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder

import pickle


# # Loading Dataset 

# In[28]:


data = pd.read_csv('Dataset.csv')


# In[29]:


data.head()


# In[30]:


data["Accident_Probability"].nunique()


# In[31]:


data.info()


# # Data Distribution

# In[32]:


df= pd.DataFrame(data)
category_counts = df['Accident_Probability'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Category Distribution of Accident Severity')
plt.show()


# # Dropping Unnecessary Columns

# In[ ]:





# In[33]:


data.head()


# # Handling Missing Values

# In[34]:


for column in data.columns:
    if data[column].dtype == 'object': 
        data[column] = data[column].fillna(df[column].mode()[0])  


# In[35]:


data.head()


# # Encode Categorical Variables

# In[36]:


# Applying Ordinal Encoding to categorical columns (excluding 'Time')
categorical_columns_final = data.select_dtypes(include=['object']).columns.tolist()
if 'Time' in categorical_columns_final:
    categorical_columns_final.remove('Time')

ordinal_encoder_final = OrdinalEncoder()
data[categorical_columns_final] = ordinal_encoder_final.fit_transform(data[categorical_columns_final])

# Creating mapping for encoded categories
category_mapping_final = {}
for col, categories in zip(categorical_columns_final, ordinal_encoder_final.categories_):
    category_mapping_final[col] = {index: label for index, label in enumerate(categories)}

# Changing data type of numeric columns
numeric_columns_final = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
data[numeric_columns_final] = data[numeric_columns_final].apply(pd.to_numeric, downcast='float')



# Displaying the first few rows of the transformed DataFrame and the category mappings
data.head(), category_mapping_final


# In[37]:


data.head(30)


# # Spliting the Dataset

# In[38]:


X = data.drop('Accident_Probability', axis=1)
y = data['Accident_Probability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[39]:


X_test


# # Model Training

# In[40]:


models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    # Create a pipeline with scaling for models sensitive to feature scaling
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy



# In[41]:


plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim([0, 1])
plt.show()


# In[42]:


results.values()


# In[43]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# # Generating Pickle File

# In[44]:


pickle.dump(dt, open('model.pkl', 'wb' ) )


# In[55]:


# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# In[56]:


# Parse input data
input_data = [2,2,2,2,2,2]



# In[57]:


# Make a prediction
prediction = model.predict([input_data])



# In[58]:


# Print the prediction as JSON
print({'prediction': prediction.tolist()})

