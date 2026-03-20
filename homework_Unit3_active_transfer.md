# Homework Unit 3: Active Learning & Transfer Learning

**Due Date:** 11:59 pm on Mar 27   
---

## 🎯 Learning Objectives
In this assignment, you will study how to learn effectively with limited labeled data. You will:

- Understand **Active Learning**: selecting the most informative samples to label and use in model training 
- Understand **Transfer Learning**: leveraging knowledge from pretrained models  
- Compare different strategies under a **low-data regime**   

---

## 📦 Dataset and Split

- Use the **Fruits & Vegetables dataset** (20 classes, 960 images total) at https://drive.google.com/file/d/1zCymISIuT3owpCKildUo5SlvccVBaDKg/view?usp=sharing 
- Suggested split:
  - **Test set:** 200 images (fixed)
  - **Validation set:** 200 images (fixed)
  - **Training pool:** remaining 560 images

### Special Setup

- **Active Learning**
  - Start with **2 labeled samples per class (40 total)**
  - Remaining data is treated as **unlabeled pool**

- **Transfer Learning**
  - Use **2 / 5 / 10 samples per class** (total: 40 / 100 / 200 labeled samples) for fine-tuning


---

## ⚙️ Setup

- Google Colab (recommended) or Local Python environment. 

### Base Model (Active Learning)
- A small CNN is sufficient  
- Train on 40 samples initially  

### Pretrained Model (Transfer Learning)
- Use an ImageNet pretrained model (e.g., MobileNetV2)

---

## 📤 Submission

Submit:

1. Google Colab link or `.ipynb`
2. Written answers (clear explanations)
3. Plots (when required)

---

# 🧪 Part 1: Active Learning (50 pts)

---

## Q1 (10 pts) — Baseline with 2 Samples per Class

Train a CNN using only **40 labeled samples** (2 samples per class). See the example code at: https://colab.research.google.com/drive/1WlX2O7Mbx__fmSnupT1TFtfvgiy4YfKO?usp=sharing 

### 📊 Plot
- Training & validation **loss curves** over training epochs
- Training & validation **accuracy curves** over training epochs

Suggesion: The number of training epochs can be set to around 40–60.

### 💭 Discussion
- Do you observe **overfitting**? If yes, explain what you observe in both the loss curves and accuracy curves.

---

## Q2 (40 pts) — Active Learning

Implement:

- **Entropy-based uncertainty sampling**
- **Random sampling baseline**

### Procedure
1. Start from  the baseline CNN model (trained on 40 samples)
2. Then   use **warm start**  continuing to train the same model as    new labeled samples are added each round. In each round:
   - Select 40 new samples
   - Add to training set
   - Continue training (warm start) the same CNN model

```python 

# ============================================================
# Active Learning: Entropy + Random Baseline (Warm Start)
# ============================================================

K_PER_ROUND = 40  # Number of samples to add per round
N_ROUNDS = 4      # Number of active learning rounds
WARM_EPOCHS = 10  # Epochs per warm-start round
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ---- Entropy-based ----
entropy_accuracies = [acc_baseline]
entropy_train_sizes = [len(initial_labeled_indices)]
labeled_entropy = list(initial_labeled_indices)
unlabeled_entropy = list(initial_unlabeled_indices)
model_entropy = build_cnn_model(no_classes)
model_entropy.set_weights(baseline_weights)

# ---- Random ----
random_accuracies = [acc_baseline]
random_train_sizes = [len(initial_labeled_indices)]
labeled_random = list(initial_labeled_indices)
unlabeled_random = list(initial_unlabeled_indices)
model_random = build_cnn_model(no_classes)
model_random.set_weights(baseline_weights)

print(f'Round 0 (baseline): train_size=40, test_acc={acc_baseline:.4f}')
print(f'Adding {K_PER_ROUND} samples per round for {N_ROUNDS} rounds')
print('=' * 70)

for round_i in range(1, N_ROUNDS + 1):

    # ===================== ENTROPY =====================
    df_ul = df_pool_all.iloc[unlabeled_entropy].copy().reset_index(drop=True)
    probs_e = get_predictions_on_pool(model_entropy, df_ul)
    entropies = compute_entropy(probs_e)
    top_k = np.argsort(entropies)[-K_PER_ROUND:]
    selected = [unlabeled_entropy[i] for i in top_k]
    labeled_entropy.extend(selected)
    unlabeled_entropy = [idx for idx in unlabeled_entropy if idx not in selected]

    reset_optimizer(model_entropy)
    df_tr = df_pool_all.iloc[labeled_entropy].copy().reset_index(drop=True)
    train_gen = create_data_generator(df_tr, shuffle=True, augment=True)
    val_gen = create_data_generator(df_val, shuffle=False)
    test_gen = create_data_generator(df_test, shuffle=False)
    model_entropy.fit(train_gen, validation_data=val_gen, epochs=WARM_EPOCHS,
                      callbacks=[early_stop], verbose=0)
    pred = model_entropy.predict(test_gen, verbose=0)
    acc_e = accuracy_score(test_gen.classes, np.argmax(pred, axis=1))
    entropy_accuracies.append(acc_e)
    entropy_train_sizes.append(len(labeled_entropy))

    # ===================== RANDOM =====================
    selected = list(np.random.choice(unlabeled_random, size=K_PER_ROUND, replace=False))
    labeled_random.extend(selected)
    unlabeled_random = [idx for idx in unlabeled_random if idx not in selected]

    reset_optimizer(model_random)
    df_tr = df_pool_all.iloc[labeled_random].copy().reset_index(drop=True)
    train_gen = create_data_generator(df_tr, shuffle=True, augment=True)
    val_gen = create_data_generator(df_val, shuffle=False)
    test_gen = create_data_generator(df_test, shuffle=False)
    model_random.fit(train_gen, validation_data=val_gen, epochs=WARM_EPOCHS,
                     callbacks=[early_stop], verbose=0)
    pred = model_random.predict(test_gen, verbose=0)
    acc_r = accuracy_score(test_gen.classes, np.argmax(pred, axis=1))
    random_accuracies.append(acc_r)
    random_train_sizes.append(len(labeled_random))

    print(f'Round {round_i}: size={len(labeled_entropy)}, '
          f'Entropy={acc_e:.4f}, Random={acc_r:.4f}')

print('\nActive Learning complete!')

```




### 📊 Plot
- X-axis: number of training samples  used in each round
- Y-axis: test accuracy  
- Compare:
  - Entropy sampling
  - Random sampling  

### 💭 Discussion (Important)

Answer thoughtfully:

- Does uncertainty sampling always outperform random?
  - If yes, please explain why.  
  - If not, why might uncertainty sampling fail?
- What is the **best accuracy achieved** by uncertainty sampling and random sampling, and when?
- Do you observe accuracy drops after adding more data?
  - Why could adding data hurt performance on test data?


---

# 🧪 Part 2: Transfer Learning (30 pts)

---

## Q3 — Pretraining + Fine-tuning

Use a pretrained model (e.g., MobileNetV2, see the example code below).

```python
## Get the pre-trained model
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
## fixing the pretrained model, which should not be trained (updated) in the classifier training process
pretrained_model.trainable = False

## define the size of input to the classifier, 224*224*3.
## Each image has 224*224 pixels. Each pixel is presented by using a combination of three colors, namely Red, Green, Blue
## This size is the same as the input of the pre-trained model
inputs = pretrained_model.input

## define the classifier, including two hidden layers, each with 128 hidden units
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

## define the output layer,  with 20 hidden units beause there are K=20 classes
outputs = tf.keras.layers.Dense(K, activation='softmax')(x)  ## K=20, the number of classes

## specify the classification model, training loss and training optimizer
model_with_pre_trained = tf.keras.Model(inputs=inputs, outputs=outputs)

model_with_pre_trained.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

## train the classifier.
## NOTE: here only the two-layer classifier is trained. The pre-trained model is not trained.
history = model_with_pre_trained.fit(
    train_images,
    validation_data=validation_images,
    batch_size = 32,
    epochs=5
)

## Get the prediction on the testing images
pred = model_with_pre_trained.predict(test_images)
pred = np.argmax(pred,axis=1)
```



Evaluate the performance on the test set under the following fine-tuning settings:
- 2 samples per class (40 total)
- 5 samples per class (100 total)
- 10 samples per class (200 total)

### 📊 Plot
- X-axis: number of  samples  used in fine-tuning
- Y-axis: test accuracy  

### 💭 Discussion

- How  does performance improve as data increases during fine-tuning? 

---

# 🧠 Part 3: Summary & Reflection (20 pts)

Answer the following:
 
- Compare Active Learning and Transfer Learning. Which method better solves the fruit and vegetable classification task under limited labeled data, and why? Support your answer with observations from your experiments.
 
- When would you prefer:
  - Active learning over transfer learning?
  - Transfer learning over active learning?

- What are the **limitations** of each method?
 

- What surprised you most in this assignment?
 