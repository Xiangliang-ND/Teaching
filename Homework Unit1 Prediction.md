# Homework Unit1 Prediction: LR vs RF vs Small MLP vs LLM Baseline

**Due Date:**  11:59 pm on Jan 30, Friday   
**Task type:** Binary classification of  Adult Income (predict whether income is `>50K`)

---

##  Learning Objectives
Use *one* tabular prediction problem to understand how different predictors behave. You will train **Logistic Regression (LR)**, **Random Forest (RF)**, **Small MLP**, and prompt an **LLM baseline** on the *same* dataset and *same* split, then compare: 
- **Predictive performance** (accuracy and F1-score) 
- **Interpretability**, when the model supports
 
---

## Dataset and Model Implementation

### 1) Recommended download (OpenML via scikit-learn)

- You can get the dataset in sklearn:
  - `fetch_openml(name="adult", version=2, as_frame=True)` (recommended)
- Target label is typically a column named `class` with values like `>50K` and `<=50K`.

> If you cannot use OpenML via scikit-learn, you may use the UCI Adult dataset files (`adult.data`, `adult.test`).
 

###  2) Data Split (Standard)
To ensure a fair comparison across models, use the same fixed split for all models:
  Train / Validation / Test = **70% / 15% / 15%**.
That means all models are trained and tested on the same dataset (The validation dataset could be used for hyperparameter selection if needed). You could  fix the random_state  when using the train_test_split funciton.

### 3) Model Implementation 
- You are  allowed to use an LLM (ChatGPT, Gemini, etc) to help generate code (e.g., ChatGPT or other coding assistants). Please  disclose whether you used, and which LLM is used for code generation/debugging. 
- The final submission must be fully runnable and reproducible (end-to-end). 
 
---

## Model 1: Logistic Regression (LR)
### Q1 -  Performance:  please report on the test set
- Accuracy
- F1 score

### Q2 -  Interpretability: please report
  - Top 3 **positive** influential features: the features with the largest positive coefficients in the logistic regression model (these push predictions toward >50K). 
  - Top 3 **negative**  influential features: the features with the most negative coefficients (these push predictions toward <=50K). 
  
  Hint: Use the learned logistic regression coefficients (after preprocessing/one-hot encoding) and sort them by value.
  
  
### Q3 - Discussion: 
Please discuss whether the top positive and negative features are reasonable. Are they consistent with common expectations about income (e.g., education, occupation, capital gain, hours worked)? 

---

## Model 2: Random Forest (RF)
Implementation suggestions: if the model performance is not good enough, you may want to tune  hyperparameters on the validation dataset, e.g.:
  - `n_estimators ∈ {200, 500}`
  - `max_depth ∈ {None, 10, 20}`
  - `min_samples_leaf ∈ {1, 5, 10}`

### Q4 -  Performance:  please report on the test set
- Accuracy
- F1 score

### Q5 -  Interpretability: please report
  - Top 3 **positive** influential features: the features with the largest positive coefficients in the logistic regression model (these push predictions toward >50K). 
  -  Please discuss whether the top positive features are the same as those reported by LR model? if different, which one makes more sense

 

## Model 3: Small MLP (Neural Network)
You may implement the MLP as a small Neural Network:
- 2–3 hidden layers
- Total hidden units ≤ ~512 (examples: 128-64, 256-128-64)
- Use ReLU (or GELU if PyTorch) as the activation function
- Use **early stopping**  based on validation to avoid overfitting, or set  Maximum epochs = 50 

### 6.3 What to report
Same metrics as LR/RF (val + test) plus:
- A learning curve plot (validation F1 or loss vs epoch)
- Brief commentary: overfitting? underfitting? how did early stopping behave?

### Q6 -  Training process monitoring
- Please shoud a learning curve plot (Accuracy or loss vs epoch on the validation dataset), like the image below. 
![alt text](image.png)

### Q7 -  Performance:  please report on the test set
- Accuracy
- F1 score

---

##  LLM Baseline: “LLM as a Tabular Model”
Goal: Use an LLM (e.g., ChatGPT, Gemini) to classify the same task by turning each row into a structured text prompt.
Prompt: Please design the prompt based on your expeirence of using LLMs. You can select some training samples as examples, and  convert each of them  to a stable, consistent schema (key=value per line).

### 7.1 Evaluation subset (Cost Control)
Because LLM inference can be expensive:
- Evaluate on **N = 200** test examples sampled randomly with a fixed seed (e.g., 42).
- You must publish the sampling seed and show the sampled indices.

 

### Q8 -  Prompt:  please report   the Prompt you used, and which LLM you tested as a baseline  



### Q9 -  Performance:   Because LLM inference can be expensive, You can choose sample 30-50 samples from the test set you used for LR, RF, and NN, and report:  
- Accuracy
- F1 score

### Q10 -   Comparison. Please discucss the performance of these models, and ...


## Reflection 

