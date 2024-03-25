# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
data = pd.read_csv('Dataset.csv')

# Display initial data info
print(data.head())
print(data.info())

# Data Distribution
category_counts = data['Accident_Probability'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Category Distribution of Accident Severity')
plt.show()

# Handling Missing Values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])

# Encode Categorical Variables
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
ordinal_encoder = OrdinalEncoder()
data[categorical_columns] = ordinal_encoder.fit_transform(data[categorical_columns])

# Change data type of numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, downcast='float')

# Split the Dataset
X = data.drop('Accident_Probability', axis=1)
y = data['Accident_Probability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

# Model Comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim([0, 1])
plt.show()

# Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Generate Pickle File
pickle.dump(dt, open('model.pkl', 'wb'))

# Load the model for demonstration
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make a demonstration prediction
input_data = [2, 2, 2, 2, 2, 2]
prediction = loaded_model.predict([input_data])
print({'prediction': prediction.tolist()})
