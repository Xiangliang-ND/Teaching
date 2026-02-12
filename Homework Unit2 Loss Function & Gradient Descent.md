# Homework Unit 2: Loss Functions (MSE vs MAE & Cross-Entropy)

**Due Date:** 11:59 pm on Feb 19, 2026

**Task type:** Regression and Classification using different Loss Functions

---

## Learning Objectives
In this assignment, you will implement and analyze different loss functions to understand their geometric properties and optimization behaviors. You will explore:
- **Regression Losses:** The difference between **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.
- **Robustness:** How outliers affect training and why certain loss functions are more robust.
- **Classification dynamics:** How **Cross-Entropy Loss** drives the decision boundary and logits during gradient descent, especially when classes have significant overlap.

---

## Prerequisites

### Environment Setup
- **Google Colab** (recommended) or Local Python environment.
- **Libraries**: `numpy`, `matplotlib`, `torch` (PyTorch is recommended for observing gradients/logits easily).

---


## Submission

Your submission should include:
1.  **Google Colab link** (with access granted) or **Jupyter Notebook** file (`.ipynb`).
2.  **Written responses** to all questions.
3.  **Generated plots** for each experiment.

---

## Part 1: Regression – MSE vs. MAE on Clean Data (35 pts)

In this section, you will compare L2 (MSE) and L1 (MAE) loss on the "Clean" data generated below.

## Provided Materials & Standardization (Crucial)


### Standardized Data Generation Code (Regression)
Please use the following code snippet to generate your regression data.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Set Seed for Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 2. Data Generation Function
def generate_data(n_samples=100):
    # Updated range: [-1.6, 1.6]
    x = np.random.uniform(-1.6, 1.6, n_samples)
    # Target function: y = sin(1*pi*x) + 0.3*x + noise
    noise = np.random.normal(0, 0.1, n_samples)
    y = np.sin(1 * np.pi * x) + 0.3 * x + noise
    return x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)

# Generate standard datasets
X_train_np, y_train_np = generate_data(100)
X_test_np, y_test_np = generate_data(100) # Testing data

# Convert to PyTorch Tensors
X_train = torch.from_numpy(X_train_np)
y_train = torch.from_numpy(y_train_np)
X_test = torch.from_numpy(X_test_np)
y_test = torch.from_numpy(y_test_np)
```

---



**Model & Training Constraints:**
To ensure consistency, use exactly these parameters:
* **Model Architecture:** A standard MLP.
    * Input Layer: 1 neuron
    * Hidden Layer: **10 neurons** with **ReLU** activation.
    * Output Layer: 1 neuron (Linear).
* **Optimizer:** SGD (Stochastic Gradient Descent).
    * *Note:* Pass the entire dataset (`X_train`) to the model in one go (Full Batch) inside the loop for smoother curves.
* **Learning Rate:** `0.05`
* **Epochs:** `3000`

### Q1 (15 pts) — Training Process Observation
Train two separate instances of the model (re-initialize the model before each training to reset weights):
1.  **Model A:** Trained using `nn.MSELoss()`.
2.  **Model B:** Trained using `nn.L1Loss()` (MAE).

**Plot:**
* A single graph showing **Training Loss vs. Iterations (Epochs)** for both MSE and MAE.
* *Tip:* You may use a Log Scale for the Y-axis if the scales differ significantly.

**Discussion:**
* **Convergence Shape:** Compare the shape of the MSE curve vs the MAE curve. Which one looks more "curved" (exponential decay) and which one looks more "linear" (straight line)?
* **Why?** Explain this based on the gradients of the loss functions ($L2$ gradient depends on error magnitude vs $L1$ gradient is constant).

### Q2 (20 pts) — Testing Performance
Evaluate both models on the **Test Set** (`X_test`, `y_test`).

**Report:**
Fill in the table:

| Model (Trained with) | Test MSE | Test MAE |
| :--- | :--- | :--- |
| **Model A (MSE)** | | |
| **Model B (MAE)** | | |

**Discussion:**
* Are the results similar?
* Which training loss resulted in smaller error on the test set? Why do you think that is?

---

## Part 2: Regression – The Effect of Outliers (35 pts)

Now you will test robustness. You must modify the training data in a **fixed** way so everyone has the same outliers.

**Data Modification (Fixed):**
Run the following code to create `y_train_noisy`:
```python
y_train_noisy = y_train.clone()
# Force the first 5 points to be outliers
y_train_noisy[0:5] = y_train_noisy[0:5] + 7.0
```
*Note: We keep the Test Set clean (unmodified).*

### Q3 (15 pts) — Training with Outliers
Retrain your models on `X_train` and `y_train_noisy` using the **exact same hyperparameters** as Part 1 (SGD, lr=0.05, 3000 epochs).
1.  **Model C:** Trained with MSE Loss.
2.  **Model D:** Trained with MAE Loss.

**Plot:**
* A scatter plot showing the data points (X vs Y).
* Highlight the **5 outlier points** in a different color.
* Draw the **prediction line** of Model C (MSE).
* Draw the **prediction line** of Model D (MAE).

### Q4 (20 pts) — Robustness Analysis
Evaluate Model C and Model D on the **Clean Test Set**.

**Report:**

| Model (Trained on Noisy Data) | Test MSE (on Clean Test) | Test MAE (on Clean Test) |
| :--- | :--- | :--- |
| **Model C (MSE)** | | |
| **Model D (MAE)** | | |

**Discussion:**
* Visually (from the plot in Q3) and numerically (from the table in Q4), which model was "pulled" more towards the outliers?
* Explain why MSE is more sensitive to outliers than MAE.

---

## Part 3: Classification – Cross-Entropy, Logits & Decision Boundary (30 pts)

In this section, you will visualize how Cross-Entropy loss drives the learning process, even when the task is difficult.

**Standardized Classification Data:**
Use this exact code to generate your 3-class dataset. 
```python
from sklearn.datasets import make_blobs
# Fixed random state for consistent clusters
X_cls_np, y_cls_np = make_blobs(n_samples=300, centers=3, cluster_std=8.0, random_state=42)

# Convert to Tensor
X_cls = torch.from_numpy(X_cls_np).float()
y_cls = torch.from_numpy(y_cls_np).long()
```

**Model & Training Constraints:**
* **Architecture:** Input (2) -> Hidden (10, ReLU) -> Output (3). **No Softmax in the model** (use `nn.CrossEntropyLoss` which applies Softmax internally).
* **Optimizer:** SGD.
* **Learning Rate:** `0.01`
* **Epochs:** `500`.



### Q5 (5 pts) — Training Curves
Train the model for 500 epochs.
**Plot:**
* Training Loss vs. Iterations.
* Classification Accuracy vs. Iterations.
* *Note:* Since the classes overlap significantly, do not expect the accuracy to reach 100%.

### Q6 (10 pts) — Decision Boundary Evolution 
You will visualize how the model learns to separate the classes over time. Plot the decision boundary at **three specific stages** using the helper function above:
1.  **Initial Step:** Before training starts (Epoch 0).
2.  **Middle Step:** Halfway through training (e.g., Epoch 50).
3.  **End Step:** After training finishes (Epoch 500).

**Discussion:**
* Describe how the boundary changes. Does it start random? Does it become more defined?
* Notice the shape of the boundary. Is it a perfectly smooth curve, or does it look like it's made of straight lines (polygonal)? Why? (Hint: Consider the ReLU activation function).


### Helper Function: Visualization
Use the following function to visualize the decision boundary. Copy this into your notebook.

```python
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    
    model.eval()
    with torch.no_grad():
        logits = model(grid_tensor)
        preds = torch.argmax(logits, dim=1)
    
    Z = preds.reshape(xx.shape).numpy()
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k', cmap=plt.cm.Spectral, alpha=0.8)
    plt.title(title)
    plt.show()
```

### Q7 (10 pts) — Logit Trajectories
Track the **3 raw logits** for specific data points indices `[0, 1, 2]` during training (e.g., every 10 epochs).

**Plot:**
Produce **3 separate plots**.
* In each plot, show 3 curves (Logit Class 0, Logit Class 1, Logit Class 2).
* **Bold** or color the curve that corresponds to the **Ground Truth** label for that point.

### Q8 (5 pts) — Logit Analysis
Observe the plots from Q7.

**Discussion:**
* **Correct Predictions:** For a point that was eventually classified correctly, does the logit of the true class keep increasing relative to the others?
* **Ambiguous/Hard Points:** Given the high overlap (`std=8.0`), describe what happens to the logits for a "hard" example compared to an "easy" one.

* **Why the Gap?** Why does Cross-Entropy loss try to push the true class logit higher even if the prediction is already correct?



