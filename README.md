# Predictive Analytics for Canine Disease Prevention: A Machine Learning Approach

## Overview
This project applies machine learning techniques to predict serious adverse health events in dogs using historical health records, symptoms, and demographic data. The goal is to enable early intervention and improve canine health outcomes through an interpretable and actionable prediction system.

## Features
- Data collection from the FDA Adverse Event Reporting System (FAERS) for veterinary drugs (2020-2024)
- Comprehensive data preprocessing and cleaning
- Feature engineering including time-based, breed, and symptom-related features
- Implementation of multiple classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors, Naive Bayes
- Model evaluation using accuracy, precision, recall, F1-score, ROC AUC, confusion matrices
- Selection of the best performing model (Random Forest) based on performance metrics
- Model explanation via feature importance visualization
- Deployment of a web-based prediction interface using Streamlit
- Ethical considerations and data privacy maintained throughout

## Technologies & Tools
- Python 3.9.7
- scikit-learn
- pandas & numpy
- Streamlit
- Snowflake Data Cloud
- SQL
- Git for version control

## Installation
1. Clone the repository:
```bash
git clone <repo_url>

1. Install dependencies:

pip install -r requirements.txt

Usage

- Fetch and preprocess data

- Train models using provided scripts

- Deploy the Streamlit app:

streamlit run app.py

Project Structure

/
├── data/                    # Raw and processed datasets
├── notebooks/               # Analysis notebooks
├── src/
│   ├── data_preprocessing.py   # Data cleaning and feature engineering scripts
│   ├── model_training.py       # Model training and evaluation scripts
│   ├── model_explanation.py    # Feature importance and explanation functions
│   └── app.py                  # Streamlit app for inference
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file

Results & Performance

- Best model: Random Forest with F1-score of 0.836, ROC AUC of 0.934

- Top predictive features: Treatment history, medical status, seizure symptoms, breed, age, weight ratios

Future Directions

- Expand to include multimodal data such as imaging and genetic info

- Improve explainability with advanced AI techniques

- Incorporate real-time data updates for dynamic predictions

- Broaden breed coverage and clinical validation

Ethical Considerations

- Data anonymization and privacy maintained

- Model intended as a decision support tool, not a standalone diagnostic system

- Bias mitigation and responsible AI practices adhered to

Acknowledgments

Inspired by ongoing developments in veterinary informatics and machine learning research.

License

This project is open-source. See LICENSE file for details.

Feel free to customize `<repo_url>` and any other sections as needed!
