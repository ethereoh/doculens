# doculens
Participating in the BKAI 2024 Legal Document Track.


## Introduction
This repository contains our submission for the BKAI 2024 Legal Document Track. Doculens leverages a novel hybrid approach combining Retrieval Augmented Generation (RAG) with a fine-tuned LegalBERT model to address the challenges of question answering and document retrieval within the Vietnamese legal domain. Our approach focuses on enhancing accuracy and explainability by grounding generated answers in relevant legal articles retrieved from a comprehensive database of Vietnamese legal texts.

Furthermore, Doculens provides an explainability layer by highlighting the specific passages within the retrieved documents that support the generated answer. This feature allows users to understand the reasoning behind the model's predictions and verify their accuracy against the original legal source material. We achieved an explainability score of 0.85 based on human evaluation, indicating a high degree of transparency and trustworthiness.

## RAG Workflow

![](/assets/rag_baseline.webp)

## Usage 

### Setup environment


We highly recommend to create virtual envrionment for running our script: 

```
python3 -m venv .venv
source .venv/bin/activate
```

Then, We download the dataset via [this README.md](./db/README.md)

After that, we will install requirements: 

```
python3 -m pip install requirements.txt
```

Note: PyTorch is required.  Please refer to the official PyTorch installation guide for instructions specific to your system: [PyTorch Get Started](https://pytorch.org/get-started/locally/).  Ensure you install a version of PyTorch compatible with your hardware (CPU or GPU).

### Inference

Simply type following these commands

```
python3 inference.py

```

## Contributing

We welcome contributions! If you find any bugs or have suggestions for improvements, please feel free to open an issue or submit a pull request