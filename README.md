

# FunctionGemma Model Evaluation

This repository contains a **local evaluation framework** for testing **Gemma 3 (2B / 4B)** instruction-tuned models on **function calling (tool use)** using `llama.cpp`.
It is designed to measure **tool selection accuracy**, **hallucination behavior**, and **response correctness** in a controlled, reproducible setup.

The project simulates a lightweight “agent” environment where the model must decide **when to call a function** versus **when to respond conversationally**.

---

## Key Features

* ✅ Local inference using **GGUF models** via `llama.cpp`
* ✅ Python-based **tool calling simulation** (no OpenAI API)
* ✅ Automatic extraction and execution of model-generated tool calls
* ✅ Mixed test suite:

  * Tool-appropriate queries
  * Casual conversation
  * Hallucination-prone prompts
* ✅ Structured test results exported to JSON
* ✅ Designed for **SLM / edge / on-device evaluation**

---

## Project Structure

```
FUNCTIONGEMMA-MODEL-EVALUATION/
│
├── model/
│   ├── functiongemma-270m-it-BF16.gguf
│   └── gemma-3-4b-it-q4_0.gguf
│
├── test_functiongemma_270m.py     # Test script for Gemma 270M
├── test_gemma3_4b.py              # Test script for Gemma 3 4B
│
├── LLAMA_CPP_CUDA_SETUP.md        # CUDA setup notes for llama.cpp
├── README.md                      # Project documentation
├── LICENSE
├── .gitignore
```

---

## Supported Tool Functions

The model is given access to the following **Python functions** and must decide when to use them:

### Utility / Info

* `get_weather(city)`
* `get_stock_price(symbol)`

### Food & Beverage

* `make_coffee(types_of_coffee, milk, sugar, strength)`
* `cook_burger(cook)`
* `cook_fries(type_of_fries)`
* `cook_prawn_noodles(prawn, sotong)`

The functions are **real Python functions**, not mock JSON calls.

---

## How Tool Calling Works

1. The system prompt injects **Python-style function definitions**
2. The model is instructed to call tools **only** inside a fenced block:

```text
```tool_code
get_weather(city="Tokyo")
````



3. The evaluator:
   - Extracts the function name + arguments
   - Executes the function
   - Verifies correctness against expectations

This mimics real-world **agent orchestration logic** without requiring an external API.

---

## Test Suite Design

Each test case specifies:
- User query
- Expected behavior:
  - Function call (and which one)
  - OR normal conversation

### Example Test Cases

| Category | Example |
|--------|--------|
| Tool use | “What’s the weather like in Tokyo?” |
| Tool use | “Give me a medium burger” |
| Conversation | “Hello! How are you?” |
| Conversation | “Tell me a joke about NVIDIA” |

This allows detection of:
- ❌ Over-calling tools
- ❌ Hallucinated function usage
- ❌ Wrong function selection

---

## Running the Evaluation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv-functiongemma
source venv-functiongemma/bin/activate
````

### 2. Install dependencies

```bash
pip install llama-cpp-python yfinance requests
```

> If using CUDA, ensure `llama-cpp-python` is built with GPU support
> (see `LLAMA_CPP_CUDA_SETUP.md`)

---

### 3. Run a test

**Gemma 3 4B**

```bash
python test_gemma3_4b.py
```

**FunctionGemma 270M**

```bash
python test_functiongemma_270m.py
```

---

## Output & Metrics

During execution, the script prints:

* Model output
* Detected function calls
* Execution results
* Inference latency per test

At the end, results are saved as:

```text
gemma3_4b_test_results.json
```

### Summary Metrics

* Total tests
* Passed / failed count
* Success rate (%)
* Per-test latency

---

## Why This Matters

This evaluation framework helps answer questions like:

* Can a small model **reliably decide when tools are needed**?
* Does the model hallucinate function calls?
* How does performance degrade from 4B → 270M?
* Is the model suitable for **edge orchestration** or **offline agents**?

It is especially useful for:

* SLM benchmarking
* Edge AI demos
* Function-calling reliability testing
* Hallucination analysis

---

## Notes & Limitations

* This is **prompt-based tool calling**, not native function calling
* Argument parsing is intentionally strict
* No retries or auto-correction logic is applied (by design)
* Weather API key is hardcoded for testing only

---

## License

This project is licensed under the terms in the `LICENSE` file.

---
