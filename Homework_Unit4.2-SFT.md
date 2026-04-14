# HW2 Unit 4:  Supervised Fine-Tuning GPT-2 model

**Due Date:** 4/18, 11:59pm   

## Goals
In this homework, you will  
   - Understand **Supervised Fine-Tuning (SFT)**: turning a base language model (e.g., GPT-2) into an instruction-following model
   - Understand **Data Preparation for SFT**: formatting instruction–response pairs for training
   - Understand **LoRA (Low-Rank Adaptation)**: a parameter-efficient fine-tuning method
   - Compare **full parameter fine-tuning vs. LoRA** in terms of trainable parameters, training efficiency, and output quality

**Note:** GPT-2 is a small model (117M parameters) trained on web text. After fine-tuning on 1,000 examples, you should see a clear behavioral shift as the model will begin following the instruction format instead of continuing the prompt randomly. However, the output quality will not match large modern models like ChatGPT; that gap is expected and part of what you will reflect on in Question 5.


## Setup

Install dependencies:

```bash
pip install transformers datasets peft trl accelerate
```

**Important:** Recommend to use at least a **T4 GPU** runtime in Google Colab, you can change that via Runtime -> Change runtime type -> T4 GPU

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### Test Prompts

Use the following 5 prompts throughout the assignment to compare model behavior before and after fine-tuning:

```python
test_prompts = [
    "Explain what machine learning is in simple terms.",
    "Write a short poem about the ocean.",
    "What are three benefits of regular exercise?",
    "Summarize the concept of supply and demand in economics.",
    "Give me a recipe for a simple pasta dish.",
]
```

### Prompt Template and Generation Helper

When generating responses, format each test prompt using this template. This template must match the format used in your training data (Q2).

```python
prompt_template = "### Instruction:\n{instruction}\n\n### Response:\n"

def generate_responses(model, tokenizer, prompts, max_new_tokens=200):
    """Generate responses for a list of prompts and print them."""
    model.eval()
    results = []
    for prompt_text in prompts:
        formatted = prompt_template.format(instruction=prompt_text)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(response)

        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_text}")
        print(f"{'-'*60}")
        print(f"RESPONSE: {response}")

    return results
```

---


### Question 1 — Baseline: GPT-2 Before Fine-tuning (10 pts)

Load the pre-trained GPT-2 model and tokenizer:

```python
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model_baseline = AutoModelForCausalLM.from_pretrained(model_name).to(device)
```

Generate responses to all 5 test prompts using `generate_responses(model_baseline, tokenizer, test_prompts)`.

### Discussion
- Show the generated output for each of the 5 test prompts.
- Does GPT-2 follow the instructions? Does it generate relevant answers or just continue the text?
- Why does this pre-trained GPT-2 language model behave this way? (think about what GPT-2 was trained to do)

---

### Question 2 — Formatting Instruction–Response Training Data (15 pts)

Here, we will use a dataset available on HuggingFace Datasets. Load the Stanford Alpaca dataset and sample 1,000 examples for training:

```python
dataset = load_dataset("tatsu-lab/alpaca")
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
```

Each Alpaca example has three fields:
- `instruction`: the task description (always present)
- `input`: optional additional context (can be empty string `""`)
- `output`: the desired response

**Implement** a `format_instruction(example)` function that converts each example into a single training string and returns `{"text": formatted_string}`. Only include `{input}` if it is not empty in the dataset. The formatted string should follow this template:

```
### Instruction:
{instruction}

### Input:          
{input}

### Response:
{output}
```

**Hint:** Check whether `example["input"]` is an empty string to decide whether to include the `### Input:` section. The `### Instruction:` and `### Response:` markers must match the `prompt_template` used in the generation helper above.

Apply your function:

```python
train_dataset = train_dataset.map(format_instruction)
```

### Discussion
- Show 3 formatted training examples (include at least one with a non-empty `input` field and one without).
- Why do we use special markers like `### Instruction:` and `### Response:` to separate the prompt from the answer?
- What is the model learning to do during SFT that it wasn't doing before?

---

### Question 3 — Fine-Tuning All Parameters with SFTTrainer (25 pts)

Fine-tune GPT-2 on the 1,000 Alpaca examples using **full fine-tuning** (all parameters are updated).

Load a fresh copy of GPT-2 (not the baseline from Q1):

```python
model_full = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
```

Report the total number of parameters (note: in full fine-tuning, all parameters are trainable, so these two numbers will be the same, and we will compare this against LoRA in Q4):

```python
total_params = sum(p.numel() for p in model_full.parameters())
trainable_params = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

**Configure** `SFTConfig` and `SFTTrainer` to train the model. Use the following suggested hyperparameters:

```python
sft_config = SFTConfig(
    output_dir="./gpt2-sft-full",
    max_seq_length=512,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model_full,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    args=sft_config,
)

import time
start = time.time()
trainer.train()
print(f"Training time: {time.time() - start:.1f}s")
```

After training, generate responses to the same 5 test prompts.

**Hint:** You can extract the training loss from the trainer's log history for plotting:

```python
log_history = trainer.state.log_history
steps = [entry["step"] for entry in log_history if "loss" in entry]
losses = [entry["loss"] for entry in log_history if "loss" in entry]
```

### Plot
- Training loss curve over training steps.

### Discussion
- Report the total number of **trainable parameters**.
- Show the generated output for each of the 5 test prompts after fine-tuning.
- Compare the outputs before (Q1) and after fine-tuning. What has changed? Does the model now follow instructions?
- What does the loss curve tell you about the training process, does the training loss converge? 

---

### Question 4 — Parameter-Efficient Fine-Tuning with LoRA (30 pts)

Now fine-tune GPT-2 using **LoRA** instead of updating all parameters.

### Step 1: Apply LoRA to a fresh GPT-2 model

Load a **fresh** copy of GPT-2 (not the fine-tuned one from Q3). Then configure and apply LoRA:

```python
model_lora = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

lora_config = LoraConfig(
    r=8,                        # Rank of the low-rank decomposition matrices A and B
    lora_alpha=32,              # Scaling factor α (update is scaled by α/r)
    lora_dropout=0.1,           # Dropout applied to LoRA layers
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn"],  # GPT-2's attention projection layer
)

model_lora = get_peft_model(model_lora, lora_config)
model_lora.print_trainable_parameters()
```

### Step 2: Fine-tune with SFTTrainer

**Configure** `SFTConfig` using the same settings as Q3, but with two key differences:
- `output_dir`: use `"./gpt2-sft-lora"`
- `learning_rate`: use `2e-4` (LoRA typically benefits from a **higher learning rate** than full fine-tuning, think about why, considering that matrix $B$ is initialized to zero at the start of training)

Create an `SFTTrainer` and train the model. Record the training time (use the same `time.time()` approach as Q3), then generate responses to the same 5 test prompts.

### Plots
- Training loss curve for LoRA (you may overlay the full fine-tuning loss from Q3 for comparison).
- A **bar chart** comparing the number of trainable parameters: full fine-tuning (Q3) vs. LoRA (Q4).

### Discussion
- Report the number of **trainable parameters** and the **percentage** of total parameters that are trainable.
- Show the generated output for each of the 5 test prompts and compare with the results from full fine-tuning (Q3). Is there a noticeable quality difference?
- Compare the number of trainable parameters between full fine-tuning (Q3) and LoRA (Q4). Using the LoRA formulation from the lecture ($h = W_0 x + \frac{\alpha}{r} B A x$), explain why there is a difference.

---

### Question 5: Summary & Reflection (20 pts)

Answer the following:

- **Before vs. After SFT:** Summarize the key behavioral differences you observed between the base GPT-2 model and the fine-tuned models. What does SFT teach the model to do that pre-training alone does not?

- **Full Fine-Tuning vs. LoRA:** Compare the two approaches across number of trainable parameters, training time, and output quality. When would you choose one over the other? 

- **Limitations of SFT:** What are some potential issues or limitations of supervised fine-tuning? Consider data quality, overfitting (especially with only 1,000 examples), and the gap between mimicking training examples vs. truly understanding instructions.