import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Canine Disease Predictor",
    page_icon="🐕",
    layout="wide"
)

# Title and introduction
st.title("🐕 Canine Disease Predictor")
st.markdown("""
This platform uses machine learning to predict the likelihood of serious adverse events in dogs based on their health data.
Enter the information below to get a prediction for your canine patient.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To", ["Prediction Tool", "Exploratory Analysis", "Model Insights", "About"])

# Load model function
@st.cache_resource
def load_models():
    try:
        # Load the models from saved files
        serious_ae_model = joblib.load('best_serious_ae_model.pkl')
        return serious_ae_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Function to preprocess input data for prediction
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Calculate age-weight ratio
    input_df['age_weight_ratio'] = input_df['ANIMAL_WEIGHT'] / (input_df['ANIMAL_AGE'] + 0.1)
   
    # Create age and weight categories
    input_df['AGE_CATEGORY'] = pd.cut(
        input_df['ANIMAL_AGE'],
        bins=[0, 1, 3, 7, 100],
        labels=['Puppy', 'Young', 'Adult', 'Senior']
    )
    
    input_df['WEIGHT_CATEGORY'] = pd.cut(
        input_df['ANIMAL_WEIGHT'],
        bins=[0, 5, 15, 30, 100],
        labels=['Small', 'Medium', 'Large', 'Extra Large']
    )
    
    # Process reactions
    input_df['REACTIONS'] = input_df['REACTIONS'].fillna('')
    common_reactions = ['vomiting', 'lethargy', 'anorexia', 'seizure', 'diarrhea', 
                       'agitation', 'panting', 'pain', 'pruritus', 'death']
    
    for reaction in common_reactions:
        input_df[f'HAS_{reaction.upper()}'] = input_df['REACTIONS'].str.contains(
            reaction, case=False, na=False).astype(int)
    
    input_df['NUM_REACTIONS'] = input_df['REACTIONS'].apply(
        lambda x: len(x.split(', ')) if x and isinstance(x, str) else 0
    )
    
    # Dummy date columns
    date_columns = [
        'ONSET_DATE_YEAR', 'ONSET_DATE_MONTH', 'ONSET_DATE_DAY',
        'ORIGINAL_RECEIVE_DATE_YEAR', 'ORIGINAL_RECEIVE_DATE_MONTH', 'ORIGINAL_RECEIVE_DATE_DAY',
        'DAYS_TO_REPORT'
    ]
    for col in date_columns:
        input_df[col] = 0  # Default value since we don't have dates in the app
     
    # Handle medical status (create dummy columns as in training)
    medical_status_options = [
        'Recovered with Sequela', 'Recovered/Normal', 'Outcome Unknown',
        'Ongoing', 'Euthanized'
    ]
    for status in medical_status_options:
        # Create a new column for each status and convert to int
        input_df[f'MEDICAL_STATUS_{status.upper().replace(" ", "_")}'] = (
            (input_df['MEDICAL_STATUS'] == status).astype(int)
        )

    top_breeds = [
        'Other',
        'Dog (unknown)',
        'Poodle (unspecified)',
        'Crossbred Canine/dog',
        'Chihuahua',
        'Poodle - Standard',
        'Beagle',
        'Maltese',
        'Boxer (German Boxer)',
        'Poodle - Miniature'
    ]
    
    # Initialize all breed columns to 0
    for breed in top_breeds:
        col_name = f'BREED_{breed.replace(" ", "_").replace("-", "_")}'
        input_df[col_name] = 0
    
    # Map selected breed to the correct column
    breed_mapping = {
        "Other/Unknown": 'BREED_Other',
        "Dog (unknown)": 'BREED_Dog_(unknown)',
        "Poodle (unspecified)": 'BREED_Poodle_(unspecified)',
        "Crossbred Canine/dog": 'BREED_Crossbred_Canine/dog',
        "Chihuahua": 'BREED_Chihuahua',
        "Poodle - Standard": 'BREED_Poodle___Standard',
        "Beagle": 'BREED_Beagle',
        "Maltese": 'BREED_Maltese',
        "Boxer (German Boxer)": 'BREED_Boxer_(German_Boxer)',
        "Poodle - Miniature": 'BREED_Poodle___Miniature'
    }
    
    selected_breed = input_df['SELECTED_BREED'].iloc[0]
    if selected_breed in breed_mapping:
        breed_column = breed_mapping[selected_breed]
        input_df[breed_column] = 1
    
    # Ensure all required columns are present
    required_columns = [
        'age_weight_ratio',
        'TREATED_FOR_AE',
        'ANIMAL_BREED_IS_CROSSBRED',
        'ANIMAL_AGE',
        'ANIMAL_WEIGHT',
        'NUM_REACTIONS'
    ] + [f'HAS_{r.upper()}' for r in common_reactions]
    
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value for missing columns

    return input_df

# Function to get prediction
def get_predictions(input_df, serious_model):
    try:
        if serious_model is not None:
            # Get probability of serious adverse event
            serious_proba = serious_model.predict_proba(input_df)[:, 1][0]
            serious_pred = 1 if serious_proba > 0.5 else 0
            return serious_pred, serious_proba
        return None, None
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def show_exploratory_analysis():
    st.header("Exploratory Analysis")
    
    # Display Treated and Serious AE
    st.subheader("Serious and Treated Adverse Event")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            image = Image.open('serious_ae_distribution.png')
            st.image(image, caption='Serious Adverse Event Distribution')
        except:
            st.info("Serious Adverse Event Distribution not available.")
    
    with col2:
        try:
            image = Image.open('treated_ae_vs_serious_ae.png')
            st.image(image, caption='Serious vs Treated Adverse Event Analysis')
        except:
            st.info("Serious vs Treated Adverse Event Analysis not available.")

    # Display Breed and Reactions
    st.subheader("Breed and Reactions Freequency")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            image = Image.open('breed_frequency.png')
            st.image(image, caption='Breed Frequency Analysis')
        except:
            st.info("Breed Frequency Analysis not available.")
    
    with col2:
        try:
            image = Image.open('reaction_types_frequency.png')
            st.image(image, caption='Reaction Types Frequency Analysis')
        except:
            st.info("Reaction Types Frequency Analysis not available.")

# Display Age weigght distributiom metrics
    st.subheader("Age and Weight Distribution")
    
    # Try to load the evaluation metrics image
    try:
        image = Image.open('age_weight_distribution.png')
        st.image(image, caption='Age and Weight Distribution')
    except Exception as e:
        st.info(f"Age and Weight Distribution visualization not available: {e}") 
    
    # Display model evaluation metrics
    st.subheader("Model Feature Importance Analysis")
    
    # Try to load the evaluation metrics image
    try:
        image = Image.open('serious_ae_feature_importance.png')
        st.image(image, caption='Model Feature Importance Analysis')
    except Exception as e:
        st.info(f"Model Feature Importance Analysis visualization not available: {e}")

# Function to display model insights
def show_model_insights():
    st.header("Model Performance Insights")
    
    # Display model evaluation metrics
    st.subheader("Model Comparison")
    
    # Try to load the evaluation metrics image
    try:
        image = Image.open('serious_ae_model_comparison.png')
        st.image(image, caption='Model Performance Comparison')
    except Exception as e:
        st.info(f"Model comparison visualization not available: {e}")
    
    # Display confusion matrices
    st.subheader("Confusion Matrices")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            image = Image.open('serious_ae_confusion_matrix_random_forest.png')
            st.image(image, caption='Serious AE Confusion Matrix (Random Forest)')
        except:
            st.info("Serious AE confusion matrix not available.")
        try:
            image = Image.open('serious_ae_confusion_matrix_k-nearest_neighbors.png')
            st.image(image, caption='Serious AE Confusion Matrix (k-nearest Neighbors)')
        except:
            st.info("Serious AE confusion matrix not available.")
    
    with col2:
        try:
            image = Image.open('serious_ae_confusion_matrix_decision_tree.png')
            st.image(image, caption='Serious AE Confusion Matrix (Decision Tree)')
        except:
            st.info("Serious AE confusion matrix not available.")
        try:
            image = Image.open('serious_ae_confusion_matrix_logistic_regression.png')
            st.image(image, caption='Serious AE Confusion Matrix (Logistic Regression)')
        except:
            st.info("Serious AE confusion matrix not available.")    
            

    with col3:
        try:
            image = Image.open('serious_ae_confusion_matrix_gradient_boosting.png')
            st.image(image, caption='Serious AE Confusion Matrix (Gradient Boosting)')
        except:
            st.info("Serious AE confusion matrix not available.")
        try:
            image = Image.open('serious_ae_confusion_matrix_naive_bayes.png')
            st.image(image, caption='Serious AE Confusion Matrix (Naive Bayes)')
        except:
            st.info("Serious AE confusion matrix not available.")            
    
    st.header("Best Model Insights")
    
    # Display model evaluation metrics
    st.subheader("Best Model ROC")
    
    # Try to load the evaluation metrics image
    try:
        image = Image.open('serious_ae_roc_curve_random_forest.png')
        st.image(image, caption='Best Model Comparison')
    except Exception as e:
        st.info(f"Model comparison visualization not available: {e}")

# Function to display about page
def show_about():
    st.header("About This Platform")
    
    st.markdown("""
    ## Predictive Analytics for Canine Disease Prevention
    
    This platform is developed to apply machine learning techniques to predict and prevent canine diseases.
    
    ### How It Works
    
    1. **Data Collection**: We collect comprehensive health data on dogs, including medical history, 
       physical characteristics, and treatment outcomes.
    
    2. **Machine Learning Analysis**: Several classification algorithms were trained on historical data 
       to identify patterns associated with serious adverse events.
    
    3. **Prediction Generation**: The best-performing model is used to predict the likelihood of 
       serious adverse events for new dog patients based on their health data.
    
    ### Data Sources
    
    Downloaded data from the open-source link below:
    
    https://open.fda.gov/data/downloads/

    The data was then loaded into the Snowflake cloud environment and connected to the model.
    
    ### Model Accuracy
    
    The platform uses advanced classifiers with:
    - Accuracy: ~85%
    - Precision: ~78%
    - Recall: ~82%
    
    ### Intended Use
    
    This tool is designed to assist veterinarians in making informed decisions about preventive care 
    for canine patients. It should be used as a complement to professional veterinary judgment, not as 
    a replacement.
    """)
def main():
    # Load the models
    serious_ae_model = load_models()
    
    if page == "Prediction Tool":
        st.header("Enter Dog Health Information")
        
        with st.form("dog_health_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                animal_species = st.selectbox("Species", ["Canine"])
                # animal_species = st.text_input("Species", value="Canine", disabled=True)
                animal_gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
                animal_reproductive_status = st.selectbox("Reproductive Status", ["Intact", "Neutered", "Mixed", "Unknown"])
                animal_age = st.number_input("Age (years)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
                animal_weight = st.number_input("Weight (kg)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
                crossbred = st.checkbox("Is the dog a crossbreed?")
                
            with col2:
                health_assessment = st.selectbox("Health Assessment Prior to Exposure", 
                                              ["Unknown", "Good", "Fair", "Poor","Excellent","No Attending Veterinarian","Critical"])
                medical_status = st.selectbox("Current Medical Status", 
                            ["Select Medical Status","Recovered with Sequela", "Recovered/Normal", "Outcome Unknown",
                                "Ongoing", "Euthanized"])
                treated_for_ae = st.checkbox("Treated for Adverse Event")
                
            # Reactions section
            st.subheader("Observed Symptoms")
            reactions = st.text_area("Enter observed reactions separated by commas (e.g., vomiting, lethargy, diarrhea)", "")
            
            # Breed section
            st.subheader("Breed Information")
            breed_options = [
                "Select breed...",
                "Other",
                "Dog (unknown)",
                "Poodle (unspecified)",
                "Crossbred Canine/dog",
                "Chihuahua",
                "Poodle - Standard",
                "Beagle",
                "Maltese",
                "Boxer (German Boxer)",
                "Poodle - Miniature"
            ]

            selected_breed = st.selectbox("Select primary breed", options=breed_options)
            
            # Submit button
            submitted = st.form_submit_button("Get Prediction")
        
        if submitted:
            auto_crossbred = crossbred
            if selected_breed == "Crossbreed/Mixed":
                auto_crossbred = True
            
            input_data = {
                'ANIMAL_SPECIES': animal_species,
                'ANIMAL_GENDER': animal_gender,
                'ANIMAL_REPRODUCTIVE_STATUS': animal_reproductive_status,
                'ANIMAL_AGE': animal_age,
                'ANIMAL_WEIGHT': animal_weight,
                'ANIMAL_BREED_IS_CROSSBRED': 1 if auto_crossbred else 0,
                'HEALTH_ASSESSMENT_PRIOR_TO_EXPOSURE_CONDITION': health_assessment,
                'MEDICAL_STATUS': medical_status,
                'TREATED_FOR_AE': 1 if treated_for_ae else 0,
                'REACTIONS': reactions,
                'SELECTED_BREED': selected_breed
            }
            
            processed_input = preprocess_input(input_data)
            
            if serious_ae_model is not None:
                serious_pred, serious_proba = get_predictions(processed_input, serious_ae_model)
                
                if serious_pred is not None:
                    st.header("Prediction Results")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("Serious Adverse Event Risk")
                        if serious_pred == 1:
                            st.error("⚠️ **HIGH RISK**")
                        else:
                            st.success("✅ **LOW RISK**")
                        
                        st.metric(
                            label="Probability",
                            value=f"{serious_proba:.2%}"
                        )
                    
                    with col2:
                        st.subheader("Risk Assessment")
                        if serious_proba < 0.25:
                            st.success("**LOW RISK**: The dog has a low likelihood of experiencing a serious adverse event.")
                        elif serious_proba < 0.50:
                            st.warning("**MODERATE RISK**: There is a moderate likelihood of a serious adverse event.")
                        elif serious_proba < 0.65:
                            st.error("**HIGH RISK**: The dog has a high likelihood of experiencing a serious adverse event.")
                        else:
                            st.error("**VERY HIGH RISK**: The dog has a very high likelihood of experiencing a serious adverse event.")
                
            # Recommendations based on risk level
            st.subheader("Recommendations")
            if serious_proba < 0.25:
                st.markdown("""
                - Standard monitoring protocol
                - Routine follow-up as normally scheduled
                - No additional preventive measures required
                """)
            elif serious_proba < 0.50:
                st.markdown("""
                - Increased monitoring frequency
                - Schedule follow-up within 1-2 weeks
                - Consider alternative treatments or dosage adjustments
                - Educate owner on early warning signs
                """)
            elif serious_proba < 0.65:
                st.markdown("""
                - Intensive monitoring protocol
                - Schedule follow-up within 3-5 days
                - Consider alternative treatments or dosage adjustments
                - Provide detailed owner education on monitoring
                - Consider preventive medications or treatments
                """)
            else:
                st.markdown("""
                - Implement highest level of monitoring
                - Daily check-ins with owner
                - Consider hospitalization for observation if appropriate
                - Strongly consider alternative treatment protocols
                - Implement all available preventive measures
                """)
            
            # Contributing factors
            st.subheader("Contributing Risk Factors")
            
            # For now, we'll show some likely risk factors based on the input
            risk_factors = []
            
            if animal_age > 7:
                risk_factors.append("Advanced age (senior dog)")
            if health_assessment == "Abnormal":
                risk_factors.append("Abnormal health assessment prior to treatment")
            if medical_status == "Diseased":
                risk_factors.append("Pre-existing disease condition")
            if any(r in reactions.lower() for r in ["vomiting", "lethargy", "seizure", "diarrhea"]):
                risk_factors.append("Early symptoms of common adverse reactions")
            if animal_weight < 5:
                risk_factors.append("Very small body weight")
                
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No specific high-risk factors identified.")
            
            # Disclaimer
            st.markdown("""
            **Disclaimer**: This prediction is based on machine learning analysis of historical data and should be used as a supportive tool for clinical decision-making, not as a replacement for professional veterinary judgment.
            """)
    
    elif page == "Exploratory Analysis":
        show_exploratory_analysis()
 
    elif page == "Model Insights":
        show_model_insights()

    elif page == "About":
        show_about()

if __name__ == "__main__":
    main()