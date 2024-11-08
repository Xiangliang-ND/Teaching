# Assignment: IMDb Sentiment Classification with BERT – Exploring Feature Reduction and Layer Depth

## Assignment Overview

In this assignment, you will explore the effect of feature reduction through Principal Component Analysis (PCA) and the influence of BERT layer depth on model performance for IMDb sentiment classification. You’ll use manual tuning techniques to observe how adjusting PCA dimensions and the number of BERT layers impacts classification accuracy.

The goal is to gain insights into the effects of dimensionality reduction and layer selection on Transformer model representations, improving both your understanding of language models' internal features and your skills in feature engineering.

## Task 1: Data Selection and Preprocessing (10pt)

**Objective**: Preprocess the IMDb dataset, extract BERT features, and prepare the data for further analysis.

### Instructions

1. **Dataset Selection**: Use the IMDb dataset for this assignment. You will select only 20% of the data to speed up processing.
2. **Tokenization and Feature Extraction**:
   - Load BERT model and tokenizer.
   - Tokenize the IMDb text data, ensuring a max length of 128 tokens.
   - Extract [CLS] token representations from multiple BERT layers for sentiment classification.
   
### Code Example

```python
import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# Load IMDb Dataset (20% subset)
# ===================Your Code (Load Dataset)=============


# Tokenize text by bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(examples):
    # ================Your Code (Tokenization)==============
    

# Apply tokenization
train_subset = train_subset.map(preprocess_data, batched=True)
test_subset = test_subset.map(preprocess_data, batched=True)
```

## Task 2: Exploring the Impact of PCA Dimensions (35pt)

**Objective**: Examine the effects of reducing feature dimensions through PCA on model accuracy. By varying PCA components, you will investigate how dimensionality reduction affects feature quality for classification.

### Instructions

1. **Feature Extraction**:
   - Extract BERT features from a fixed layer (e.g., layer 6) using the [CLS] token representations.
2. **Dimensionality Reduction**:
   - Apply PCA on extracted features using several dimensional settings (e.g., 50, 100, 200).
   - Train a logistic regression model on the reduced features.
3. **Performance Evaluation**:
   - Record accuracy for each PCA dimension.
   - Plot the relationship between PCA dimensions and model accuracy on both training and testing data.

### Code Example

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to extract and reduce features
def get_reduced_features(texts, model, tokenizer, layer, pca_components):
    # ==============Your Code (Get Reduced Features)================
    
    return cls_embeddings

# Testing different PCA dimensions
pca_dimensions = [None, 50, 100, 200]
results = {}

for pca_dim in pca_dimensions:
    # ===============Your Code (Classification by LogisticRegression)====================
    

# Plotting results
# ===============Your Code (Visualization)====================
```

## Task 3: Exploring the Impact of BERT Layer Depth (35pt)

**Objective**: Analyze the effect of different BERT layers on the quality of extracted features for sentiment classification.

### Instructions

1. **Feature Extraction**:
   - Extract BERT features from multiple layers (e.g., layers 0, 2, 4, 6, etc.).
   - Use [CLS] token representations for feature extraction.
2. **Model Training**:
   - For each layer’s features, train a logistic regression model for classification.
3. **Performance Evaluation**:
   - Record the model accuracy for each layer.
   - Plot accuracy versus layer depth to determine the optimal feature extraction layer.

### Code Example

```python
# Exploring different BERT layers
layers = [0, 2, 4, 6, 8, 10, 12]
layer_results = []

for layer in layers:
    # ===============Your Code (Classification by LogisticRegression)====================
    
    print(f"Layer {layer}: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

# Plotting results
# ===============Your Code (Visualization)====================
```

## Task 4: Comparative Analysis and Discussion (20pt)

**Objective**: Reflect on how PCA and layer depth influence model performance.

- Write a short analysis discussing the effects of PCA and BERT layer selection on model performance.
- Which PCA dimension and BERT layer yielded the highest accuracy?
- Discuss the trade-offs in using lower PCA dimensions or layers for feature extraction.





## Task 5: Optional Task with Extra Credit – Space and Time Representation in Language Models (50pt)

**Objective**: For this optional task, you will explore advanced research on spatial and temporal representations in language models. Your goal is to read and analyze findings from the paper "*Language Models Represent Space and Time*" (https://arxiv.org/pdf/2310.02207) and then implement a notebook provided on Google Drive to understand these representations practically.

### Instructions

1. **Paper Reading**:

   - Read the research paper "*Language Models Represent Space and Time*."
   - Focus on understanding how language models capture information about spatial and temporal concepts.

2. **Implementation and Analysis**:

   The code for this task is in Google Drive: https://drive.google.com/drive/folders/15jzKetU0Vw7MAgvl0rkiuIK0NLgk8VL2?usp=sharing

   1. **Task 1: Code Completion**:
      - In the project’s main directory, locate and open the files:
        - `save_activations.py`
        - `probe_experiment.py`
        - `make_prompt_datasets.py`
      - Inside each file, find the lines marked `'''TODO xxxxxxxx'''`. These are placeholders where you will complete specific code segments.
      - Fill in these sections to ensure the activation-saving and probing experiments are properly implemented.
   2. **Task 2: Visualization and Pipeline Execution**:
      - Open the provided [run.ipynb](https://drive.google.com/file/d/1MFssSKAUuDLWXoxEkyKPXi2ncS0JfPMW/view?usp=sharing) file in Google Colab or your local Jupyter environment. 
      - Complete the visualization code within `run.ipynb` to interpret spatial and temporal representations.
      - Once completed, execute the entire notebook to generate the visualizations.

3. **Submission Requirement**:
   - Take a screenshot of your results after executing the notebook.

   - Submit three files for code completion: `save_activations.py`, `probe_experiment.py`, `make_prompt_datasets.py`

     

## Submission Guidelines

Submit the following items:

1. **Code**: Provide code for Tasks 1–4 in a Jupyter notebook or Python script.
2. **Results**: Include tables or figures comparing the results for each PCA dimension and BERT layer.
3. **Analysis**: Write a brief report (100–300 words) summarizing findings and observations.