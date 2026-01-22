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
That means all models are trained and tested on the same dataset (The validation dataset could be used for hyperparameter selection if needed). You could  fix the ''random_state''  when using the train_test_split funciton.

### 3) Model Implementation 
You are  allowed to use an LLM (ChatGPT, Gemini, etc) to help generate code. Please  disclose whether you used, and which LLM is used for code generation/debugging.  
 
---

## Submission
Your submission should include **runnable, end-to-end code** for LR, RF, and MLP. This can be a Jupyter notebook file (.ipynb) or a Google Colab link (with access granted).

Your LLM baseline work can be included in the same submission as text (within the notebook or as a written section), answering the corresponding questions (prompt, model used, results, discussion).

---
## Model 1: Logistic Regression (LR)
### Q1 (10pts) -  Performance:  please report on the test set
- Accuracy
- F1 score

### Q2 (10pts) -  Interpretability: please report
  - Top 3 **positive** influential features: the features with the largest positive coefficients in the logistic regression model (these push predictions toward >50K). 
  - Top 3 **negative**  influential features: the features with the most negative coefficients (these push predictions toward <=50K). 
  
  Hint: Use the learned logistic regression coefficients (after preprocessing/one-hot encoding) and sort them by value.
  
  
 - Please discuss whether the top positive and negative features are reasonable. Are they consistent with common expectations about income (e.g., education, occupation, capital gain, hours worked)? 

---

## Model 2: Random Forest (RF)
Implementation suggestions: if the model performance is not good enough (for example, accuracy is lower than 0.8), you may want to tune  hyperparameters on the validation dataset, e.g.:
  - `n_estimators ∈ {200, 500}`
  - `max_depth ∈ {None, 10, 20}`
  - `min_samples_leaf ∈ {1, 5, 10}`

### Q3 (10pts)  -  Performance:  please report on the test set
- Accuracy
- F1 score

### Q4 (10pts)  -  Interpretability: please report
  - Top 3 **positive** influential features, e.g., using the Permutation importance function.
  -  Please discuss whether the top positive features are the same as those reported by LR model? if different, which one makes more sense.

 

## Model 3: Small MLP (Neural Network)
You may implement the MLP as a small Neural Network:
- 2–3 hidden layers
- Total hidden units ≤ ~512 (examples: 128-64, 256-128-64)
- Use ReLU (or GELU if PyTorch) as the activation function
- Use **early stopping**  based on validation to avoid overfitting, or set  Maximum epochs = 50 

 

### Q5 (10pts)  -  Training process monitoring
- Please show a learning curve plot (loss vs epoch on the validation dataset), like the image below. 
![alt text](image.png)

### Q6 (10pts) -  Performance:  please report on the test set
- Accuracy
- F1 score

---

##  LLM Baseline: “LLM as a Tabular Model”
Goal: Use an LLM (e.g., ChatGPT, Gemini) to classify the same task by turning each row into a structured text prompt.

Prompt: Please design the prompt based on your expeirence of using LLMs. You can select some training samples as examples, and  convert each of them  into a structured prompt.

Because LLM inference can be expensive, you can evaluate on **30-50** test examples sampled randomly from the fixed test set you have used for LR, FR and NN.

 

### Q7 (5pts) -  Prompt + LLM used:  

Please report   
 - the form of prompt you used  
 - which LLM you tested as a baseline  (model name/version if available)
 - whether you used 0-shot or few-shot (and how many examples used in the prompt)



### Q8 (5pts) -  Performance:    
- Accuracy
- F1 score

### Q9 (10pts)  -   Comparison. 
Compare LR vs RF vs MLP vs LLM: 
- Which performs best? Which is most explainable? 
- What kinds of mistakes does the LLM make compared to classic models?

 

## Reflection (20pts) 
(required, 6–10 sentences)

If you had to deploy one model, which would you choose and why, under each of the following principles?

(a) **Interpretability-first**: Which model would you deploy if you must clearly explain predictions to stakeholders (e.g., feature importance or human-readable reasoning)? Briefly justify.

(b) **Accuracy-first**: Which model would you deploy if the primary goal is the best predictive performance on the test set? Briefly justify using your results.

(c) **Lowest cost/effort-first**: Which model would you deploy if you care most about simplicity, training/inference cost, and ease of maintenance? Briefly justify.
