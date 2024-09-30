# Homework Assignment of Unit 2: Monitoring Training and Validation Curves

**Due Date:** 11:59pm on Oct 8, Tuesday

**Submission:** Please submit your solution as a Jupyter Notebook (*.ipynb). Ensure your notebook runs without errors when executed from top to bottom. Include clear comments and discussions as needed, and print outputs as required.

## **Overview**

In this assignment, you will explore how to monitor training and validation curves while training a simple neural network model. The objective is to understand how learning curves can help identify issues such as overfitting and underfitting, and how various hyperparameters can affect model performance.

## **Assignment Tasks**

### **Part 1: Understanding Learning Curves (15 points)**

**Before answering the questions, please have a try on the following code blocks.**

**Dataset Download: https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set**

**Load your dataset:** 

```python
import pandas as pd
data = pd.read_excel("Real estate valuation data set.xlsx")
print(data.info())
data.head()
```

**Look at the data items about real estate valuation:** 

```python
# rename the columns
renamed_columns = [col.split()[0] for col in data.columns]
renamed_columns_map = {data.columns[i]:renamed_columns[i] for i in range(len(data.columns))}

data.rename(renamed_columns_map, axis=1, inplace=True)

# remove No column
data.drop("No", axis=1, inplace=True)

print(data.head())

# separate features and target data
features, target = data.columns[:-1], data.columns[-1]

X = data[features]
y = data[target]
```

**Evaluation Metric:** Root mean square error (RMSE) is widely used as a performance measure in continuous value prediction. It measures the average difference of the actual data points from the predicted values, and the difference is squared to avoid the cancelation of positive and negative values, while they are summed up.

##### a. Example 1: Decision Tree

- A model with high variance is said to be overfitting.

- For example, the decision tree regressor is a non-linear machine learning algorithm. Non-linear algorithms typically have low bias and high variance.

```python
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# overfitting
decision_tree = DecisionTreeRegressor()

train_sizes, train_scores, test_scores = learning_curve(
    estimator=decision_tree,
    X=X,
    y=y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    train_sizes = [1, 75, 165, 270, 331]
)

train_mean = -train_scores.mean(axis=1)
test_mean = -test_scores.mean(axis=1)

plt.subplots(figsize=(10,8))
plt.plot(train_sizes, train_mean, label="train")
plt.plot(train_sizes, test_mean, label="validation")

plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.legend(loc="best")
plt.show()
```

**b. Example 2: SVM**

- A model with high bias is said to be underfit.

- The support vector machine (SVM) is a linear machine learning algorithm. Linear algorithms typically have high bias and low variance.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Underfitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm = SVR(C=0.25)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=svm,
    X=X_scaled,
    y=y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    train_sizes = [1, 75, 150, 270, 331]
)

train_mean = -train_scores.mean(axis=1)
test_mean = -test_scores.mean(axis=1)

plt.subplots(figsize=(10,8))
plt.plot(train_sizes, train_mean, label="train")
plt.plot(train_sizes, test_mean, label="validation")

plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.legend(loc="best")

plt.show()
```

**c. Example 3: Random Forest**

- A good fit model exists in the gray area between an underfit and overfit model. The model may not be as good on the training data as it is in the overfit instance, but it will make far fewer errors when faced with unseen instances.

- The random forest is an ensemble of decision trees. This means the model is also non-linear, but bias is added to the model by creating several diverse models and combining their predictions.

```python
from sklearn.ensemble import RandomForestRegressor

# better
random_forest = RandomForestRegressor(max_depth=3)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=random_forest,
    X=X,
    y=y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    train_sizes = [1, 75, 150, 270, 331]
)

train_mean = -train_scores.mean(axis=1)
test_mean = -test_scores.mean(axis=1)

plt.subplots(figsize=(10,8))
plt.plot(train_sizes, train_mean, label="train")
plt.plot(train_sizes, test_mean, label="validation")

plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.legend(loc="best")

plt.show()
```

**After running all code blocks, answer the question:** 

- Describe how to identify overfitting and underfitting through training and validation curves. Provide examples or scenarios that illustrate these concepts.

### **Part 2: Implementation of a Simple Neural Network (20 points)**

**Task Description:**
Implement a simple feedforward neural network using the MNIST dataset and monitor the training and validation curves.

**Code Examples for the Steps to Follow:**

1. **Data Preparation (10 points):**
   - Load the MNIST dataset and perform necessary preprocessing (e.g., normalization, train-test split).
   - Provide a brief explanation of the dataset and any preprocessing steps taken.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

2. **Model Definition (15 points):**
   - Define a simple neural network architecture. 

```python
# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes for MNIST

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
```

3. **Training the Model (15 points):**
   - Train the model while logging the training and validation loss and accuracy for each epoch.

```python
# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Validate the model
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_losses.append(val_loss / len(test_loader))

# Plot training and validation losses
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### **Part 3: Hyperparameter Tuning (50 points)**

1. **Experiment with Hyperparameters:**
   - Adjust the following hyperparameters and observe their effects on the learning curves:
     - **Learning Rate:** Try values like 0.1, 0.01, and 0.001.
     - **Batch Size:** Experiment with sizes like 32, 64, and 128.
     - **Number of Neurons:** Modify the number of neurons in the hidden layers (e.g., 64, 128, 256).
     - **Number of Epochs:** Train for different numbers of epochs (e.g., 5, 10, 20).

**Example of changing the learning rate:**

```python
# Change the learning rate for the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Experiment with different values
```

2. **Analysis of Results:**
   - Discuss how the adjustments affected model performance. Did you observe any instances of overfitting or underfitting? How did the learning curves inform your analysis?

### **Part 4: Reflection and Summary (15 points)**

1. **Reflect on Your Experiments (select two of the following questions to answer):** Please provide detailed answers to the following questions based on your experimental observations:
   
   - **Q1: Learning Curve Analysis:** Describe how the training and validation loss curves changed with different hyperparameter settings. Did you notice any patterns? For instance:
     - What happened to the validation loss when you increased the learning rate? Did it diverge?
     - How did changing the batch size affect the training dynamics and stability of the model?
     - Did you observe any signs of overfitting or underfitting? What specific curves indicated these issues?

   - **Q2: Challenges Encountered:** Discuss specific challenges you faced during training. For example:
     - Did you experience convergence issues with certain learning rates? How did you address them?
     - Were there any particular configurations where the model did not perform as expected? What were your hypotheses about why that might have happened?

   - **Q3: Overall Observations:** Summarize your overall findings:
     - Which hyperparameter configurations yielded the best validation accuracy?
     - What were your takeaways about the relationship between hyperparameters and model performance?

2. **Submission Requirements:**
   - Write your responses to the chosen questions in a markdown cell in your Jupyter Notebook.
   - Use clear and concise language, and include visual examples or results where relevant.
   - Submit your Jupyter Notebook (*.ipynb) with all code cells executed and results visible.

## **Resources**
- [Learning Curves Tutorial: What Are Learning Curves?](https://www.datacamp.com/tutorial/tutorial-learning-curves)
- Keras Documentation: [Link](https://keras.io/)
- PyTorch Documentation: [Link](https://pytorch.org/)

