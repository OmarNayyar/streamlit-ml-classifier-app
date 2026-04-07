# Mushroom Classifier Studio

A polished Streamlit portfolio project built from the guided Coursera app **"Build a Machine Learning Web App with Streamlit and Python."**  
This version preserves the original machine learning workflow while improving the interface, code structure, documentation, and presentation for GitHub, LinkedIn, CV, and freelance portfolio use.

## Overview

This application uses the mushroom classification dataset to compare three supervised learning models:

- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest

Users can select a classifier, adjust key hyperparameters, train the model, and review evaluation metrics and plots in a clean dark-themed Streamlit interface.

## Features

- Professional dark-mode UI with a cleaner layout and stronger visual hierarchy
- Sidebar-driven model selection and parameter controls
- Built-in help text for classifiers, hyperparameters, metrics, and plots
- Stable training and evaluation flow with reusable functions
- Performance metrics using `st.metric`
- Visual diagnostics for confusion matrix, ROC curve, and precision-recall curve
- Dataset preview and feature overview tabs
- Preserved baseline file (`base_app.py`) for direct comparison with the original guided project

## Tech Stack

- Python
- Streamlit
- Pandas
- scikit-learn
- Matplotlib

## Project Structure

```text
project-root/
├── app.py
├── base_app.py
├── mushrooms.csv
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── assets/
    └── screenshots/
```

## Guided Project vs Enhancement

### What the original guided project included

- Mushroom dataset loading
- Basic classifier selection
- Hyperparameter inputs
- Training and test split
- Accuracy, precision, and recall output
- Confusion matrix, ROC curve, and precision-recall plots

### What was added in this upgraded version

- A modern, dark-themed portfolio presentation
- Cleaner function decomposition and more maintainable code
- Improved sidebar organization and help text
- Better metric presentation with `st.metric`
- Structured landing section, summary cards, and tabs
- Updated plotting flow using current scikit-learn display utilities
- Stable cached data loading and session-state based result handling
- Improved repository documentation and project packaging

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

After launch, open the local Streamlit URL shown in the terminal.

## Future Improvements

- Add cross-validation as an optional evaluation mode
- Add downloadable prediction summaries or experiment snapshots
- Add a lightweight model comparison table across multiple runs
- Deploy the app on Streamlit Community Cloud

## Author

**Omar Nayyar**  
Portfolio project upgrade based on an original Coursera guided exercise.

## License

This project uses the MIT License. See the [LICENSE](LICENSE) file for details.
