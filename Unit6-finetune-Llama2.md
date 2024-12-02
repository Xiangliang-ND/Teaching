# Assignment: Finetuning Large Language Models for Emoji Generation

## Assignment Overview

In this assignment, you will finetue a large language model (like Llama2) on a custom dataset for emoji generation. This is an example of adapting LLMs on your custom dataset for a specific task.

In the tasks of Emoji Generation, given a sentence, the model is expected to give an appropriate combination of emojis to describe the sentence. For example, given the sentence “Let’s go dancing tonight.”, the expected output of the model would be “💃🕺🎶”.

An Example Code can be found at: https://colab.research.google.com/drive/1lSoom3XcPnuKdiW_ckjTnn1BrhxBcOsr?usp=sharing


## Task 1: Custom instruction dataset (10pt)

**Objective**: Build a custom instruction dataset to fine-tune the language model.

### Instructions

The raw data is already in the example code. please convert the raw data to the instruction dataset. (Hint: How to construct a dataset in the huggingface format, please read https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html Please remember to add Llama special tokens to your dataset.)


## Task 2: Finetune Llama2 and Show the results (50pt)

**Objective**: Examine the effects of fine-tuning results.

### Instructions

1. **Evaluate the Fine-Tuned Model on Known Prompts**:
   - Test the fine-tuned model by generating emojis for five prompts that were included in the instruction dataset. Analyze how accurately and consistently the model generates the intended emojis.
2. **Evaluate the Fine-Tuned Model on Novel Prompts**:
   - Test the fine-tuned model by generating emojis for five prompts that were **NOT** included in the instruction dataset. Compare the outputs to evaluate the model's ability to generalize beyond the training data.
3. **Explore Edge Cases**:
   - Create and test five edge-case prompts that combine ambiguous or contradictory meanings (e.g., "happy and sad at the same time"). Evaluate how the fine-tuned model handles these cases, and document any notable failures or unexpected outputs.

 
## Task 3: Compare the fine-tuned model and ChatGPT (20pt)

**Objective**: Analyze the generated emojis by these two different models  for identical prompts, focusing on quality, relevance, creativity, and generalization.  

### Instructions

1. **Prompts Adaptation for ChatGPT**:
   - Use the same five prompts from Task 2.2, and adapt them for ChatGPT. To ensure ChatGPT responds with emojis rather than a text-based answer, include explicit instructions in the prompts, such as: Generate only emojis that symbolize these ideas, or Respond with emojis that capture this event.
2. **Performance Comparison**:
   - Generate emoji outputs with ChatGPT. Compare these outputs to those from your fine-tuned LLaMA 2 model, focusing on quality, relevance, creativity, and generalization. 


## Task 4: Investigating Overfitting in Fine-Tuning (20pt)

**Objective**: analyze whether increasing the number of training epochs during fine-tuning leads to overfitting and to evaluate its impact on the model’s ability to generalize.

### Instructions

1. **Increase Training Epochs**:
   - Fine-tune the model again, increasing the number of training epochs to 5.
   - Ensure you use the same custom dataset used in previous fine-tuning experiments.
2. **Evaluate for Overfitting**: 
   - Test the model by providing five emoji generation prompts and discuss the quality of generated emojis.
   - Test the model by providing five prompts that do not ask for emoji generatio. For example: Explain the concept of gravity, or What is the capital of Germany? Compare the model's responses to these prompts with the outputs from the original LLaMA 2 model (pre-fine-tuning) to identify changes in behavior and generalization. 
3. **Discuss Overfitting and Solutions**: 
   - Reflect on the results of the tests and determine if the model exhibits signs of overfitting, such as: producing irrelevant emoji-based outputs for non-emoji tasks, or showing reduced diversity or adaptability in its responses.
   - Discuss strategies to address overfitting in fine-tuning large language models. Consider whether approaches like early stopping, regularization, data augmentation, and hyperparameter tuning are reasonable solutions. Summarize your findings and reasoning.

     

## Submission Guidelines

Submit the following items:

1. **Code**: Provide code for Tasks 1–4 in a Jupyter notebook or Python script.
2. **Results**: Include promts and results for each task.
3. **Discussion**: Include the discussuon of task 4.3 (100–300 words).
