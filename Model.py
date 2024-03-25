import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    # Fill missing values for categorical data
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    # Ordinal encoding for categorical columns
    encoder = OrdinalEncoder()
    data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
    return data

# Function for splitting dataset
def split_dataset(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train models and compare their performance
def train_and_compare_models(X_train, X_test, y_train, y_test):
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
    return results

# Main execution function
def main():
    data = load_and_preprocess_data('Dataset.csv')
    X_train, X_test, y_train, y_test = split_dataset(data, 'Accident_Probability')
    results = train_and_compare_models(X_train, X_test, y_train, y_test)
    # Display results
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.show()
    # Save the Decision Tree model as an example
    dt_model = DecisionTreeClassifier().fit(X_train, y_train)
    pickle.dump(dt_model, open('dt_model.pkl', 'wb'))

if __name__ == "__main__":
    main()
