I've consolidated today's learning into a beginner-friendly, interview-focused crash revision document. It covers PEFT, LoRA, QLoRA, PyTorch Dataset, DataLoader, the training loop, code flow, and memory tricks without making it unnecessarily long.

# PEFT, LoRA, QLoRA & PyTorch — Last-Minute Interview Revision

> **Goal:** Understand the complete fine-tuning workflow from creating a dataset to training and using a LoRA adapter.

## 1. First understand the big picture

Imagine we have a pretrained LLM and want to customize it for our business.

For example:

- Understand customer feedback.
- Classify messages as positive or negative.
- Answer questions in a specific domain.

We can fine-tune the pretrained model using our own data.

**Complete flow:**

```
Raw Dataset (CSV)
       |
       v
Tokenizer
(Text -> Numbers)
       |
       v
PyTorch Dataset
(Organizes individual examples)
       |
       v
DataLoader
(Creates batches)
       |
       v
Pretrained Model + LoRA
       |
       v
Forward Pass -> Loss
       |
       v
Backpropagation
       |
       v
Optimizer updates LoRA weights
       |
       v
Save LoRA Adapter
       |
       v
Load Base Model + Adapter
       |
       v
Prediction
```

## 2. Fine-tuning vs Full Fine-tuning vs PEFT

| Concept          | Meaning                                              | Memory requirement |
| ---------------- | ---------------------------------------------------- | ------------------ |
| Pretrained model | Model already trained on large amounts of data       | Depends on model   |
| Fine-tuning      | Further training a pretrained model on specific data | Depends on method  |
| Full fine-tuning | Update all model parameters                          | High               |
| PEFT             | Train only a small number of parameters              | Lower              |

**PEFT = Parameter-Efficient Fine-Tuning.**

Instead of updating the entire model, we freeze most original parameters and train only a small set of additional or selected parameters.

**Memory trick:**

- Full fine-tuning = Update almost everything.
- PEFT = Update only a small portion.

## 3. Important PEFT methods

| Method         | Simple explanation                                              |
| -------------- | --------------------------------------------------------------- |
| LoRA           | Adds small trainable low-rank matrices to selected layers.      |
| QLoRA          | Uses a 4-bit quantized base model + LoRA adapters.              |
| Adapter Tuning | Adds small neural network layers between existing model layers. |
| Prompt Tuning  | Learns trainable prompt embeddings.                             |
| Prefix Tuning  | Learns trainable prefixes used in attention.                    |
| P-Tuning       | Uses trainable prompt representations.                          |
| IA³            | Learns scaling vectors to control activations.                  |
| BitFit         | Trains only bias parameters.                                    |

**Remember:** LoRA and QLoRA are especially important for practical LLM fine-tuning interviews.

## 4. LoRA — Low-Rank Adaptation

### What problem does LoRA solve?

Full fine-tuning updates all model weights, which requires significant memory and computation.

LoRA freezes the original model and adds small trainable matrices.

### Understand the formula

[\
W\_{\text{new}} = W + \frac{\alpha}{r}BA\
]

| Symbol    | Meaning                                        |
| --------- | ---------------------------------------------- |
| W         | Original pretrained weight matrix              |
| A and B   | Small trainable matrices                       |
| BA        | Creates the weight update                      |
| r         | Rank; controls the size/capacity of the update |
| α (alpha) | Controls the update scaling                    |

### Why is it called low-rank?

Suppose the original weight matrix is:

[\
W = 1024 \times 1024\
]

It contains:

[\
1024 \times 1024 = 1,048,576\
]

parameters.

Instead of training this entire matrix, LoRA uses two smaller matrices with rank 8:

[\
A = 8 \times 1024\
]

[\
B = 1024 \times 8\
]

Trainable parameters:

[\
8192 + 8192 = 16,384\
]

So we train only 16,384 parameters instead of 1,048,576 for this matrix.

**Important distinction:**

- Add parameter counts: A parameters + B parameters.
- Multiply matrices: BA creates the weight update.

### Important LoRA parameters

| Parameter        | Meaning                                                         |
| ---------------- | --------------------------------------------------------------- |
| `r`              | Rank; higher rank means more trainable parameters and capacity. |
| `lora_alpha`     | Controls update strength through scaling.                       |
| `lora_dropout`   | Helps reduce overfitting.                                       |
| `target_modules` | Specifies which model layers receive LoRA.                      |

Common target modules in Transformer models:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

Names depend on the model architecture. For example, DistilBERT uses `q_lin` and `v_lin`.

### How does LoRA training work?

1. Load the pretrained model.
2. Freeze original model weights.
3. Attach LoRA matrices A and B.
4. Pass training data through the model.
5. Calculate loss.
6. Calculate gradients using backpropagation.
7. Update LoRA parameters using the optimizer.
8. Save the trained adapter.

**Interview point:** Frozen base weights do not get updated, but gradients can still flow through the model's operations to train the LoRA parameters.

## 5. QLoRA — Quantized LoRA

### Why do we need QLoRA?

Even with LoRA, we still need to load the complete base model into memory.

QLoRA reduces this memory requirement by storing the frozen base model in a quantized format, commonly 4-bit.

### What is quantization?

Quantization means representing model weights using fewer bits.

| Format    | Bits per weight | Approx. storage for 7B weights |
| --------- | --------------- | ------------------------------ |
| FP32      | 32              | 28 GB                          |
| FP16/BF16 | 16              | 14 GB                          |
| INT8      | 8               | 7 GB                           |
| 4-bit     | 4               | 3.5 GB                         |

These are approximate raw weight-storage figures, not total training memory.

### QLoRA architecture

```
Pretrained Base Model
         |
         v
  4-bit Quantization
         |
         v
   Frozen Base Model
         |
         +------ LoRA Matrices A and B
                         |
                         v
                 Train LoRA Only
                         |
                         v
                 Save Adapter
```

### Three important QLoRA techniques

| Technique           | Simple explanation                                                                 | Memory trick                         |
| ------------------- | ---------------------------------------------------------------------------------- | ------------------------------------ |
| NF4                 | 4-bit NormalFloat format designed for typical neural network weight distributions. | Better-suited 4-bit representation   |
| Double Quantization | Quantizes the quantization constants too.                                          | Compress the compression information |
| Paged Optimizers    | Help manage temporary memory spikes using paging techniques.                       | Handle memory peaks                  |

**Important:** The base weights are stored in 4-bit, but computations can use higher precision such as BF16 or FP16.

### LoRA vs QLoRA

| Feature              | LoRA                  | QLoRA                             |
| -------------------- | --------------------- | --------------------------------- |
| Base model           | Usually FP16/BF16     | Usually 4-bit                     |
| Base weights         | Frozen                | Frozen                            |
| Trainable parameters | LoRA matrices         | LoRA matrices                     |
| Memory requirement   | Higher                | Lower                             |
| Main advantage       | Efficient fine-tuning | More memory-efficient fine-tuning |

**One-line memory trick:**

- LoRA = Train small updates.
- QLoRA = Compress base model + train small updates.

## 6. PyTorch Dataset — Starting from zero

### What is PyTorch Dataset?

A Dataset tells PyTorch:

1. How many examples exist?
2. How to retrieve one example?

We generally create a custom Dataset by inheriting from:

```
torch.utils.data.Dataset
```

Two important methods:

| Method               | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `__len__()`          | Returns total number of examples.        |
| `__getitem__(index)` | Returns one example at a given position. |

### Example dataset

`sentiment.csv`

```
text,label
"I love this product",1
"Very happy with service",1
"I hate this product",0
"Poor customer service",0
```

Here:

- 1 = Positive
- 0 = Negative

### Custom PyTorch Dataset

```
import torch
import pandas as pd
from torch.utils.data import Dataset

class SentimentDataset(Dataset):

    def __init__(self, csv_file, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        text = self.data.iloc[index]["text"]
        label = self.data.iloc[index]["label"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
```

### Understand tokenizer output

A neural network cannot directly process raw English sentences.

Tokenizer converts text into numerical token IDs.

```
"I love this product"
          |
          v
     Tokenizer
          |
          v
 [101, 1045, 2293, 2023, ...]
```

The Dataset returns three important things:

| Field            | Meaning                               |
| ---------------- | ------------------------------------- |
| `input_ids`      | Numerical representation of text      |
| `attention_mask` | Identifies real tokens versus padding |
| `labels`         | Correct answer for training           |

**Memory trick:** Text → Tokenizer → Numbers → Model.

## 7. What is DataLoader?

Dataset provides individual examples.

DataLoader collects those examples into batches.

```
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)
```

Suppose we have 12 examples:

```
Dataset = 12 examples

DataLoader (batch_size=4)

Batch 1 = 4 examples
Batch 2 = 4 examples
Batch 3 = 4 examples
```

| Parameter      | Meaning                                   |
| -------------- | ----------------------------------------- |
| `batch_size`   | Number of examples processed together     |
| `shuffle=True` | Randomly changes example order each epoch |

**Memory trick:**

- Dataset = Individual examples.
- DataLoader = Groups examples into batches.

## 8. LoRA training loop — Most important code

```
# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

# Calculate loss
loss = outputs.loss

# Calculate gradients
loss.backward()

# Update trainable parameters
optimizer.step()

# Clear old gradients before next batch
optimizer.zero_grad()
```

Understand every step:

| Step | Code                    | Meaning                                                              |
| ---- | ----------------------- | -------------------------------------------------------------------- |
| 1    | `model(...)`            | Forward pass; model processes input and calculates predictions/loss. |
| 2    | `outputs.loss`          | Measures prediction error.                                           |
| 3    | `loss.backward()`       | Calculates gradients.                                                |
| 4    | `optimizer.step()`      | Updates trainable weights.                                           |
| 5    | `optimizer.zero_grad()` | Clears previous gradients.                                           |

### Important training terms

| Term            | Simple meaning                                                 |
| --------------- | -------------------------------------------------------------- |
| Epoch           | One complete pass through the training dataset.                |
| Batch           | A group of training examples.                                  |
| Loss            | How wrong the model's predictions are.                         |
| Gradient        | Information about how parameters should change to reduce loss. |
| Backpropagation | Calculates gradients.                                          |
| Optimizer       | Uses gradients to update trainable parameters.                 |
| Learning rate   | Controls how big each parameter update is.                     |

**Memory trick:**

Forward → Loss → Backward → Update → Clear gradients.

## 9. Save and load LoRA adapter

### Save

```
model.save_pretrained("./sentiment_lora_adapter")
```

### Load

```
from peft import PeftModel

base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model = PeftModel.from_pretrained(
    base_model,
    "./sentiment_lora_adapter"
)
```

**Important:** A LoRA adapter usually stores only the adapter parameters and any explicitly saved task-specific modules. You generally need the matching base model to use it.

## 10. Most asked interview questions

**Q1. Why do we use PEFT?**

PEFT reduces the cost of fine-tuning by training only a small number of parameters instead of updating the entire pretrained model.

**Q2. Does LoRA update original model weights?**

No. Original model weights remain frozen. LoRA updates its additional low-rank matrices.

**Q3. Why is LoRA called low-rank adaptation?**

Because it represents the weight update using two smaller matrices, A and B, whose product creates the update.

**Q4. What is the difference between A+B and BA?**

A+B means adding their parameter counts to calculate the total trainable parameters. BA is matrix multiplication that creates the weight update.

**Q5. Does QLoRA train the 4-bit base weights?**

No. The quantized base model remains frozen. LoRA matrices are trained.

**Q6. Does 4-bit mean all calculations happen in 4-bit?**

No. The base weights are stored in 4-bit, while computations can use BF16 or FP16.

**Q7. What is the difference between Dataset and DataLoader?**

Dataset provides individual examples. DataLoader groups examples into batches.

**Q8. Why do we tokenize text?**

Because neural networks require numerical inputs rather than raw text.

**Q9. What is an epoch?**

One complete pass through all training examples.

**Q10. What happens during `loss.backward()`?**

PyTorch calculates gradients for trainable parameters. The optimizer uses those gradients to update parameters.

**Q11. What is the difference between LoRA and RAG?**

| LoRA                                           | RAG                                      |
| ---------------------------------------------- | ---------------------------------------- |
| Fine-tunes model behavior using training data. | Retrieves relevant external information. |
| Changes adapter parameters.                    | Does not normally change model weights.  |
| Useful for task/domain adaptation.             | Useful for current or private knowledge. |

They can also be combined: RAG supplies relevant information, while LoRA adapts the model's behavior.

## 11. Final 30-second revision

Remember these statements before an interview:

1. **PEFT:** Fine-tune only a small portion of model parameters.
2. **LoRA:** Freeze base model and train low-rank matrices.
3. **QLoRA:** Quantize frozen base model to 4-bit and train LoRA.
4. **NF4:** 4-bit format designed for neural network weights.
5. **Double quantization:** Quantizes quantization constants.
6. **Paged optimizer:** Helps manage memory spikes.
7. **Dataset:** Provides individual training examples.
8. **DataLoader:** Creates batches from the dataset.
9. **Tokenizer:** Converts text into token IDs.
10. **Forward pass:** Model processes inputs and produces outputs.
11. **Loss:** Measures prediction error.
12. **Backward pass:** Calculates gradients.
13. **Optimizer:** Updates trainable parameters.
14. **Epoch:** One complete pass over the dataset.
15. **Adapter:** Small trained component that can be saved separately from the base model.

### The ultimate memory formula

```
DATA -> TOKENIZE -> DATASET -> DATALOADER
                    |
                    v
              BASE MODEL + LORA
                    |
                    v
              FORWARD -> LOSS
                    |
                    v
                 BACKWARD
                    |
                    v
                 UPDATE
                    |
                    v
               SAVE ADAPTER
                    |
                    v
                 INFERENCE
```

> **Interview tip:** Don't just memorize the LoRA formula. Be able to explain why we need LoRA, why QLoRA saves more memory, how a PyTorch Dataset feeds the model, and which parameters actually get updated. If you can explain this complete flow, you understand the practical foundation of PEFT.

**My recommendation:** Revise sections 1, 4, 5, 6, 8 and 10 just before your interview. Those cover the core concepts, implementation flow and most likely follow-up questions.
