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

## Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Install Dependencies
```bash
pip install -r requirements.txt

Running the Analysis & Model Comparison

The main analysis, data exploration, feature engineering, model training, and comparison are handled by the script:

Msc_Project_code_eda_and_model_comparison.py

To execute, run:

python Msc_Project_code_eda_and_model_comparison.py

This will perform data processing, train multiple models, evaluate their performance, and output results including model metrics and visualizations.

Running the Streamlit App

The Streamlit app allows for interactive prediction and visualization. To launch the app, run:

streamlit run Msc_Project_code_streamlit_app_build.py

Project Structure

/
├── data/                           # Raw and processed datasets
├── notebooks/                      # Analysis notebooks (if any)
├── scripts/
│   ├── Msc_Project_code_eda_and_model_comparison.py  # Main analysis, model comparison
│   └── Msc_Project_code_streamlit_app_build.py        # Streamlit Web App
├── requirements.txt               # Dependencies
├── README.md                        # Documentation
└── .gitignore                       # Git ignore file

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
