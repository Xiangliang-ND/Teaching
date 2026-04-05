# Homework Unit 4: How Language Models Represent and Understand Words

**Due Date:** 11:59 pm on Apr 12   

## Goals
   -  Understand word representations from word2vec and BERT
   -  Distinguish word representations vs causal summary for generation


## Part 1: Understanding Word2Vec Representations

In the Word2Vec (CBOW or skip-gram) model, each word in the vocabulary is represented as a one-hot vector $x_i \in \mathbb{R}^V$, where $V$ is the vocabulary size. The model has two weight matrices:

- $W_1 \in \mathbb{R}^{d \times V}$ (input → hidden)
- $W_2 \in \mathbb{R}^{V \times d}$ (hidden → output)

Given a  word $x_i$, the  model (e.g., skip-gram) computes:
$
h = W_1 x_i, \quad u = W_2 h, \quad \hat{y} = \text{softmax}(u)
$

### Question 1.1 — Word Representation  by Word2Vec (10pts)

For a given word $x_i$, explain **how the Word2Vec model learns and represents this word**, step by step.
Your explanation should clearly address:
1. **Multiplication with $W_1$**
   - What is the form of $h = W_1 x_i$?
   - Why does this operation select a column of $W_1$?
   - What does $h$ represent?

2. **Training process**
   - What is the model trying to predict?
   - How does the prediction task update $W_1$?

3. **Final representation**
   - After training, where is the representation of $x_i$ stored?
   - What form does this representation take?
   - Why do similar words have similar vectors?

### Question 1.2 — Word2Vec Model Exploration
### Setup

Install dependencies:

```bash
pip install transformers torch gensim
```

Load models:

```python
import gensim.downloader as api
w2v = api.load("word2vec-google-news-300")
```

### Word Selection

Choose an ambiguous word, such as **bank**, and define:

- **Set A (meaning 1)**: river, shore, coast
- **Set B (meaning 2)**: money, loan, finance
- **Set C (unrelated)**: banana, table

### Question 1.2.1 —  Similarity of word representation vectors (5pts)

- Compute cosine similarity between *bank* and words in A, B, and C. Which words are most similar to *bank*?

```python
v_bank = w2v["bank"]   # get the vector of "bank"
``` 

### Question 1.2.2 — Does Word2Vec distinguish different meanings of *bank*? (5pts) 
- Explain using your similarity results.

### Question 1.2.3 — Similar Words in Word2Vec (5pts)
Find the top-10 most similar words to *bank* (whose vectors are closest to *bank* in the embedding space):

```python
w2v.most_similar("bank", topn=10)
```
- What meaning seems to dominate among these top-10 similar words? financial meaning or river meaning? 

---

## Part 2: Understanding BERT Representations

In BERT Transformers, we perform **self-attention**, where keys, queries, and values are all derived from the same input sequence.
Let $\{x_1, \dots, x_n\} \subset \mathbb{R}^d$ be a sequence of  input tokens. Let $W^V, W^K, W^Q \in \mathbb{R}^{d \times d}$ be learned attention matrices:
$
v_i = W^Vx_i, \quad k_i = W^Kx_i, \quad q_i = W^Qx_i.
$

### Question 2.1  — Self-attention Explantion (15pts)

For a given token $x_i$, derive its context representation vector $c_i$ **step by step**.
Your derivation must include:

1. Computation of $q_i$
2. Similarity scores:
   $
   s_{ij} = \frac{q_i^T k_j}{\sqrt{d}}
   $
3. Attention weights:
   $
   \alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{m=1}^n \exp(s_{im})}
   $
4. Final context vector:
   $
   c_i = \sum_{j=1}^n \alpha_{ij} v_j
   $

### Explain briefly

- What is the role of **query**?
- What is the role of **keys**?
- What is the role of **values**?
- What do attention weights represent?

---
### Question  2.2 — BERT Model Exploration (10pts)
Load models:
```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
```

Use context sentences such as:

```python
inancial_bank_sentence = "He deposited money in the bank."
river_bank_sentence = "The boat reached the river bank."


def get_bert_word_embedding(sentence, target_word):
    """
    Return contextual embedding for target_word in sentence.
    If target word is split into subwords, average their vectors.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]   # [seq_len, hidden_dim]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    span = find_subword_span(tokens, target_word)
    if span is None:
        raise ValueError(f"Could not find '{target_word}' in tokenized sentence: {tokens}")

    start, end = span
    word_vec = hidden[start:end+1].mean(dim=0).cpu().numpy()
    return word_vec, tokens, (start, end)


vec_fin_bank, _, _ = get_bert_word_embedding(financial_bank_sentence, "bank")
vec_riv_bank, _, _ = get_bert_word_embedding(river_bank_sentence, "bank")

print("\n[BERT] Same word, different contexts:")
print(f"  bank(financial) vs bank(river): {cos_sim(vec_fin_bank, vec_riv_bank):.4f}")

```

Extract contextual embeddings for:

- bank (financial meaning)
- bank (river meaning)

Also extract embeddings for words in A, B, and C using context sentences of your choice.

Then compute cosine similarities between:

- bank(financial) and words in A, B, C
- bank(river) and words in A, B, C
- Does BERT distinguish different meanings of *bank*?  Support your answer with similarity values.



## Part 3: Comparing Word2Vec vs BERT Representations

In this part, you will compare:
- **Word2Vec (static embeddings)**
- **BERT (contextual embeddings)**
 


### Question 3.1 — Visualization (Word2Vec) (5pts)

Visualize embeddings of the following words using **t-SNE**, and discuss  your observation
- bank
- words in A, B, C


### Question 3.2 — Visualization (BERT) (5pts)

Visualize embeddings of the following words using **t-SNE**, and discuss  your observation:

- bank (financial)
- bank (river)
- words in A, B, C

### Question 3.3 — Visualization Comparison Word2Vec vs BERT (5pts)
How Word2Vec vs BERT representations visualized differently?

---

### Question 3.4 —   Repeat for another ambiguous word: "apple" (15pts)

Define:

- **Set A (fruit)**: orange, banana
- **Set B (company)**: iphone, mac, google
- **Set C (unrelated)**: river, car

Repeat Questions 3.1–3.3 for **apple**.

---
## Part 4: Autoregressive Attention and Generation
In autoregressive language models (for example, GPT-style models), the goal is to **generate the next token given the previous tokens**.
As a result, the context vector $c_i$ serves a different role than in standard self-attention: instead of representing the input token $x_i$,  it is a causal summary of past tokens that is used to predict the next token $x_{i+1}$.
 
Consider the following two partial sentences:

1. **“He deposited money in the”**
2. **“The boat reached the river”**

In both cases, the next word could be **“bank”**, but the meaning of *bank* is different.


### Question 4.1 —   Masked self-attention Illustration (15pts)
Draw a figure showing how $c_i$ is computed for the input: **He deposited money in the**.  Let the input tokens be  $x_1, x_2, \dots, x_i$.
Your figure should:

- show the input tokens:   "He", "deposited", "money", "in", "the"
- show which previous tokens the current position can attend to,
- show how the attended information is combined into $c_i$,
- show that $c_i$ is then used to predict the next word **“bank.”** (You only need to draw one layer of masked self-attention (you can ignore stacked Transformer layers. You may hand-draw the figure and submit it as an image.)

### Question 4.2 —  How Context is relevant? (5pts)
For this input  $x_1, x_2, \dots, x_i$ =  **“The boat reached the river”**. How the resulting context vector $c_i$ is   different from the one in Q4.1? Provide your discussion (no need diagram drawing).

---
 ## Optional Bonus Question (20 pts)

Are you curious about how the GPT context vector $c_i$ are similar or different when having these two partial inputs:   **He deposited money in the** and   **The boat reached the river**? 
This bonus question is intended to help you explore how an autoregressive language model (like GPT) forms a context-dependent representation before predicting the next token.

Explore the code of a GPT model at https://colab.research.google.com/drive/1SVgEvyANiiayOcYL7OY4kJjxPcPOyzdr?usp=sharing  

Based on your results, discuss:
   - Are the two $c_i$ vectors similar or different? and WHY?
   - Try changing the partial contexts to other examples of your choice, and explain your observations.

