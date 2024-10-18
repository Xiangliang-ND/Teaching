### Assignment Overview

In this assignment, you will explore how to manually tune hyperparameters and use AutoML techniques to automatically optimize parameters such as learning rate, batch size, and number of epochs. The goal is to understand how these parameters affect model performance and compare manual tuning with AutoML-based optimization.

You will experiment with both approaches to see how hyperparameter optimization can impact the model's accuracy, training time, and overall performance.

---

### Task 1: Data Selection and Preprocessing (10pt)

**What you need to do**: In this task, you will choose a dataset, preprocess it, and prepare it for machine learning model training. 

1. **Select a Dataset**: You can use any dataset from sources such as the UCI Machine Learning Repository or Kaggle. For example, you could choose the *Iris*, *Breast Cancer*, or *MNIST* dataset for this assignment.
   
2. **Preprocess the Data**:
    - Handle missing values (if any) using appropriate imputation methods.
    - Scale the features to ensure better convergence (e.g., using StandardScaler).
    - Split the data into training and test sets.

#### Code Example:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset (e.g., UCI's Iris dataset)
df = pd.read_csv('path_to_your_dataset.csv')

# Features (X) and labels (y)
X = df.drop(columns='target_column')
y = df['target_column']

# Add your code: Split into training and test sets (you can also standardize the features)
################################################################################

```

---

### Task 2: Manual Hyperparameter Tuning (30pt)

**What you need to do**: For this task, you will manually adjust several key hyperparameters for a machine learning model of your choice. You must run multiple experiments to observe the impact of each hyperparameter on model performance. Manually adjust the following hyperparameters for two simple machine learning models: Neural Networks and Random Forest.

For each combination of hyperparameters, train the model and record the accuracy, and loss on test set. You should try at least 3 combinations. In the end, please repoort the best combination setting for the hyperparameters of each model, along with the corresponding accuracy and loss. 

#### Code Example:

Some short examples for Neural Networks parameter tuning:

- **Learning Rate**: Experiment with values like 0.001, 0.01, 0.1.
- **Batch Size**: Try batch sizes of 16, 32, 64.
- **Number of Epochs**: Train for 10, 50, and 100 epochs.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Build the model
def create_model(learning_rate):
    # Add your code: define your model
    ################################################################################
    return model

# Manually tune hyperparameters
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
epochs = [10, 50, 100]

# Train the model with a specific set of hyperparameters
model = create_model(learning_rate=0.01)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')
```

**For Random Forest tuning, experiment with different values of the parameters:**

Some short examples for Random Forest parameter tuning (use sklearn):

- **`n_estimators`**: 50, 100, 200, 500 (Number of trees).
- **`max_depth`**: 5, 10, 20, None (Maximum tree depth).
- **`min_samples_split`**: 2, 5, 10 (Minimum samples to split a node).
- **`min_samples_leaf`**: 1, 2, 4, 10 (Minimum samples in a leaf node).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Manually tune Random Forest parameters
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy}')
```

---

### Task 3: AutoML for Hyperparameter Tuning (40pt)

**What you need to do**: In this task, you will use an AutoML tool to automatically optimize the hyperparameters you manually tuned in Task 2. You will compare the modal performance with auto-tuned hyperparameters against that with manual tuning. 


You can use AutoML libraries such as Auto-sklearn (not work well in Colab), TPOT, and Optuna. Take the following example codes for reference.


#### AutoML Code Example (TPOT):

```python
# Step 1: Import necessary libraries
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score

# Step 2: Initialize TPOTClassifier
# The generations and population_size determine how long the algorithm will run.
# You can adjust these parameters based on your computational resources.
tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)

# Step 3: Fit TPOT to the training data
tpot.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = tpot.predict(X_test)

# Step 5: Evaluate the TPOT model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'TPOT Model Accuracy: {accuracy}')

# Step 6: Export the best model pipeline
# TPOT will automatically generate and save the best pipeline code it found.
tpot.export('best_tpot_pipeline.py')
```


#### AutoML Code Example (Optuna):

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define objective function for optimization
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best parameters
print(study.best_params)
```




### Task 4: Comparative Analysis and Discussion (20pt)

**What you need to do**: After completing both manual and AutoML-based tuning, you will compare the results and reflect on the efficiency of each approach. Analyze the strengths and weaknesses of each method.

1. **Compare the Results**:
    - Record the accuracy, loss, and training time for both manual tuning and AutoML.
    - Present the results in a table or chart comparing the accuracy and time for each method.
2. **Analysis**:
    - Write a short analysis discussing which method was more efficient (manual tuning or AutoML).
    - Discuss the strengths and weaknesses of each approach.

### Submission Guidelines

Submit the following items:
1. **Code**: Provide the code for both manual and AutoML-based hyperparameter tuning (Jupyter notebook or Python script).
2. **Results**: Include tables/graphs comparing the results for each hyperparameter combination.