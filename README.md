# Pretrained Encoders Spurious Learning Analysis

This repository contains the code and analysis for our project on evaluating the behavior of pretrained language models when fine-tuned on datasets with spurious correlations. Our focus is on understanding how models like BERT and RoBERTa handle spurious features and how counterfactual data augmentation can mitigate these effects.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

Pretrained language models have achieved remarkable success across various NLP tasks. However, they can be susceptible to spurious correlations present in training data, leading to poor generalization. This project investigates:

- The extent to which pretrained models rely on spurious features.
- The effectiveness of counterfactually augmented data (CAD) in reducing spurious correlations.
- The impact of token sequence limits on model performance.

Our analysis includes experiments on datasets like Toxic Spans and employs metrics to quantify the reliance on spurious features.

## Project Structure

```
.
├── finetuning/
│   └── Scripts for fine-tuning models on datasets.
├── CAD_Metrics.ipynb
├── CAD_Metrics_No_Ratio.ipynb
├── Counterfactually_Augmented_Data_(CAD)_Analysis_and_Dataset_Preparation.ipynb
├── DecompX_Colab_Demo.ipynb
├── Final_Report.pdf
├── Token_Sequence_Limits.ipynb
├── Toxic_Spans_Analysis.ipynb
├── Toxic_Spans_Metrics.ipynb
├── Visualizations.ipynb
├── toxic_phrases.ipynb
├── training_stats.csv
└── README.md
```

- `finetuning/`: Scripts for fine-tuning BERT and RoBERTa on relevant datasets.
- `*_Metrics.ipynb`: Notebooks for computing CAD and Toxic Spans-specific metrics.
- `DecompX_Colab_Demo.ipynb`: Demonstrates DecompX usage for interpretability.
- `Final_Report.pdf`: Contains detailed findings and conclusions.
- `Visualizations.ipynb`: Plots and visual summaries of results.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip

### Installation

```bash
git clone https://github.com/metehanberker1/pretrained-encoders-spurious-learning-analysis.git
cd pretrained-encoders-spurious-learning-analysis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Fine-tuning Models

Navigate to the `finetuning/` directory and run:

```bash
python finetune_model.py
```

### Run Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Open and run any of the `.ipynb` notebooks to replicate analyses or view results.

## Results

Key findings:
- Pretrained models often rely heavily on spurious features.
- CAD improves robustness and reduces reliance on non-generalizable signals.
- Token sequence limits significantly impact model behavior.

More details can be found in the relevant notebooks and in [Final_Report.pdf](./Final_Report.pdf).

## Citation

If this project helps your work, please consider citing:

```
@misc{pretrained_encoders_spurious_learning,
  author = {Metehan Berker, Charley Sanchez, Nathan Huey},
  title = {Pretrained Encoders Spurious Learning Analysis},
  year = {2024},
  howpublished = {\url{https://github.com/metehanberker1/pretrained-encoders-spurious-learning-analysis}}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
