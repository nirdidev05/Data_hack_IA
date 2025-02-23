# 🔍 DataHack IA - TinyPy Language Model

Welcome to the **DataHack IA** repository, where we tackle the challenge of **AI-driven programming problem-solving** using a transformer-based model trained on Python-like code.

---

## 🚀 **Project Overview**

This project was developed as part of a **Data Science & AI competition** on **Kaggle**, aiming to build an advanced **TinyPy Language Model** capable of reasoning through **if statements, loops, arithmetic expressions, and variable updates**.

### 🔥 **Challenge Description**
The task is to create an AI model that can:
- **Understand and track conditional (`if`, `elif`, `else`) execution.**
- **Correctly simulate variable updates within loops (`while`, `for`).**
- **Perform arithmetic calculations with memory augmentation.**
- **Handle multi-step execution flow and maintain context.**

The core objective was to **train a GPT-based model** to solve Pythonic programming logic **without losing track of variable state**.

---

## 🏗 **Project Architecture**

The solution consists of **three main components**:

### 1️⃣ **Data Preparation**
- Tokenization with a **custom TinyPy tokenizer** that encodes common Python tokens (`if`, `while`, `+`, `print(...)`, etc.).
- Dataset conversion into **binary tokenized format (`train.bin`, `val.bin`, `test.bin`)** for efficient training.
- Execution step tracking **(`STEP_ID`)** to maintain variable memory across iterations.

### 2️⃣ **Training Script (ddp_datahack.py)**
- **Distributed Training** with **PyTorch DDP (torchrun)**.
- **Transformer Architecture** inspired by **GPT** with **Memory-Augmented Attention**.
- **Gradient Clipping & Weight Decay** to prevent exploding gradients.
- **Cosine Learning Rate Decay** for optimized convergence.
- **Checkpoints & Logging** for tracking training loss and evaluation loss.

### 3️⃣ **Evaluation Script (eval-script.py)**
- **Loads the trained model** (`best-model.pth`) and runs it on test data.
- **Measures accuracy** of generated predictions against expected outputs.
- **Ensures variable tracking consistency** in `if`, `while`, and arithmetic operations.

---

## 📊 **Training Details**
- **Model:** GPT with **8 layers, 384 embedding dim, 8 heads**
- **Batch Size:** 190
- **Training Steps:** ~1 epoch
- **Loss Curve:**
  - **Initial loss:** `~5.10`
  - **Final loss:** `~0.02`
  - **Overfitting Prevention:** **Dropout, weight decay, and careful step scheduling.**

---

## 🛠 **How to Set Up & Run the Project**

### 🏗 **1. Setup Virtual Environment**
```sh
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
🎯 Challenge Hosted On Kaggle
🔗 GitHub Repo: DataHack IA
💡 Feel free to contribute by improving the tokenizer, training efficiency, or adding new reasoning tasks to enhance the AI model!