# Skinup AI

![Skinup AI System Architecture](assets/diagram/skinup_architecture.png)

## Introduction

Skinup AI is a minimum viable product (MVP) that analyses a skin image, identifies a likely skin condition, scores skincare products by relevant ingredients, retrieves supporting ingredient information, and uses a local large language model (LLM) to turn the results into a user-friendly report.

The project was inspired by a practical problem in beauty retail: many beauty advisers do not have enough specialist knowledge to confidently match skincare products to a customer's skin condition. Skinup AI is intended to act as an assistant that helps an adviser make a more informed product selection after the image has been analysed. It supports—not replaces—human judgement.

This is a learning project with basic functionality, not a production system. Retrieval-augmented generation (RAG) was introduced primarily for learning purposes and out of curiosity about how the architecture works in practice.

> [!IMPORTANT]
> Skinup AI is not a medical device and does not provide a diagnosis or medical advice. Its output may be incomplete or incorrect. Do not use it to diagnose or treat a condition, and consult a qualified healthcare professional for persistent, severe, or concerning symptoms. Always verify current product labels and patch-test products before use.

### How it works

1. The user supplies the path to a local skin image.
2. OpenCV checks basic resolution, brightness, and image clarity requirements.
3. A Hugging Face image-classification model predicts the most likely skin condition.
4. Products are scored according to ingredients associated with that condition.
5. ChromaDB retrieves ingredient evidence from the local knowledge base.
6. A Hugging Face instruction-tuned LLM explains the recommendations in plain language.

### MVP functionality

- Local command-line workflow
- Basic image-quality validation
- Skin-condition image classification (Oily, acne, dry, and normal skin condition)
- Ingredient-based product scoring
- Local ChromaDB retrieval for ingredient context
- Locally generated recommendation explanations

The MVP does not include user accounts, authentication, an HTTP API, a graphical interface, clinical validation, demographic fairness evaluation, production monitoring, or guaranteed recommendation accuracy. However, the system outputs have been reviewed by a beauty industry expert, who confirmed that the ingredient information used in the recommendations is accurate.

### Data and models

The product data used during development was downloaded from Kaggle. The data files are intentionally excluded from Git, so a local copy must be placed in the paths configured in `config/config.json`. The exact source dataset identifier and its licence are not currently recorded in the repository; these should be added here before the dataset or a derived product catalogue is redistributed.

The skin-analysis and language models are downloaded from Hugging Face on first use:

- Skin analysis: [`tuphamdf/skincare-detection`](https://huggingface.co/tuphamdf/skincare-detection)
- Report generation: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

Review and comply with each dataset and model card, licence, acceptable-use terms, and known limitations. Product names, ingredients, URLs, and formulations can change; recommendations must be checked against the current product label.

## Development

### Requirements

- Python 3.10 or later (several pinned dependencies require Python 3.10+)
- Enough storage to download both Hugging Face models and build the local vector database
- Sufficient RAM/VRAM for the 4B-parameter LLM; `device_map="auto"` selects available hardware
- Internet access for the initial model downloads

### Setup

Clone the repository, create an isolated environment, and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local `.env` file containing the public Hugging Face model identifiers:

```env
SKIN_MODEL="tuphamdf/skincare-detection"
LLM="Qwen/Qwen3-4B-Instruct-2507"
```

The model identifiers are public configuration values, not API secrets. Do not add Hugging Face access tokens or other credentials to source files, logs, or commits. The `.env` file is ignored by Git.

### Local data

The following configured files are required but excluded from the repository:

```text
data/skincare_products/products.csv
data/skincare_products/cleaned_common_ingredients_knowledge_v3.csv
data/skincare_products/aliases.json
```

The product CSV is expected to include at least `product_name` and `ingredients`. The ingredient knowledge CSV must match the fields read by `rag/ingest.py`, including `ingredient`, `condition`, `recommendation_role`, evidence fields, and source metadata.

To build or refresh the local ChromaDB collection:

```bash
python -m rag.ingest
```

To run the application:

```bash
python main.py
```

Enter a path to a local image when prompted. Model files remain in the Hugging Face cache, and inference is performed locally by Transformers. The image itself is not intentionally uploaded by this code.

### Repository structure

```text
main.py                    CLI orchestration
config/config.json         global project settings
dataset/products.py        product dataset preprocessing
model/skin_analyser/       image-classification wrapper
model/llm/                 report-generation wrapper
rag/                       ChromaDB ingestion and retrieval
utils/                     image validation and product scoring
```

## Conclusion

Skinup AI demonstrates an end-to-end learning workflow that connects image classification, ingredient-based scoring, RAG, and local LLM generation. As an MVP, it intentionally focuses on the smallest set of useful features and on understanding how these components work together. Before any real-world or customer-facing use, the system needs stronger data provenance, security controls, clinical and fairness evaluation, product-data maintenance, testing, and review by qualified skincare or healthcare professionals.
