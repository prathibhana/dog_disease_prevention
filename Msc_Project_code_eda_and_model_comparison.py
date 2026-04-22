import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import snowflake.connector
import time
import joblib

# Snowflake connection parameters
snowflake_params = {
    'user': 'PRATHIBHANA',
    'password': 'Choosetowin&#1',
    'account': 'cihrlul-yr35747',
    'warehouse': 'COMPUTE_WH',
    'database': 'MSC_DB',
    'schema': 'RAW_DATA'
}

# Function to connect to Snowflake
def connect_to_snowflake(params):
    try:
        conn = snowflake.connector.connect(
            user=params['user'],
            password=params['password'],
            account=params['account'],
            warehouse=params['warehouse'],
            database=params['database'],
            schema=params['schema']
        )
        print("Successfully connected to Snowflake!")
        return conn
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

# Function to fetch data from Snowflake - focusing only on columns needed for SERIOUS_AE prediction
def fetch_dog_health_data(conn):
    try:
        # Query with columns needed for SERIOUS_AE prediction
        query = """
        SELECT 
            ORIGINAL_RECEIVE_DATE, 
            ONSET_DATE, 
            SERIOUS_AE, 
            TREATED_FOR_AE, 
            ANIMAL_SPECIES, 
            ANIMAL_GENDER, 
            ANIMAL_REPRODUCTIVE_STATUS, 
            ANIMAL_AGE_MIN AS ANIMAL_AGE, 
            ANIMAL_WEIGHT_MIN AS ANIMAL_WEIGHT, 
            ANIMAL_BREED_IS_CROSSBRED, 
            HEALTH_ASSESSMENT_PRIOR_TO_EXPOSURE_CONDITION, 
            MEDICAL_STATUS,
            REACTIONS, 
            BREEDS
        FROM MSC_DB.RAW_DATA.DOG_HEALTH_RECORDS_FACT
        """
        dog_health_df = pd.read_sql(query, conn)
        print(f"Successfully fetched {len(dog_health_df)} records!")
        return dog_health_df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Data preprocessing function with focus on SERIOUS_AE
def preprocess_data(df):
    # Make a copy to avoid modifying the original data
    processed_df = df.copy()
    
    # Display missing values
    print("\nMissing values per column:")
    missing_values = processed_df.isnull().sum()
    missing_percent = (missing_values / len(processed_df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
    print(missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False))
    
    # Basic statistics
    print("\nBasic statistics for numerical columns:")
    print(processed_df.describe().T)

    # Feature engineering section
    def engineer_features(df):
        # Make a copy of the dataset
        df_engineered = df.copy()
        
        # Age-weight ratio (may indicate health issues if abnormal)
        if 'ANIMAL_AGE' in df_engineered.columns and 'ANIMAL_WEIGHT' in df_engineered.columns:
            # Avoid division by zero
            df_engineered['age_weight_ratio'] = df_engineered['ANIMAL_WEIGHT'] / (df_engineered['ANIMAL_AGE'] + 0.1)
        
        return df_engineered
    
    # Apply feature engineering
    processed_df = engineer_features(processed_df)

    # Handle date columns
    date_columns = ['ORIGINAL_RECEIVE_DATE', 'ONSET_DATE']
    for col in date_columns:
        if col in processed_df.columns:
            # Convert to datetime
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            
            # Extract useful features from dates where dates are not null
            processed_df[f'{col}_YEAR'] = processed_df[col].dt.year
            processed_df[f'{col}_MONTH'] = processed_df[col].dt.month
            processed_df[f'{col}_DAY'] = processed_df[col].dt.day

    # Calculate days between dates
    if 'ORIGINAL_RECEIVE_DATE' in processed_df.columns and 'ONSET_DATE' in processed_df.columns:
        # Initialize the column with zeros
        processed_df['DAYS_TO_REPORT'] = 0
        
        # Only calculate for rows where both dates are valid
        valid_mask = (~processed_df['ORIGINAL_RECEIVE_DATE'].isna()) & (~processed_df['ONSET_DATE'].isna())
        
        # Apply computation only to valid rows
        for idx in processed_df[valid_mask].index:
            try:
                # Explicitly convert both to pandas Timestamp to ensure compatibility
                receive_date = pd.Timestamp(processed_df.at[idx, 'ORIGINAL_RECEIVE_DATE'])
                onset_date = pd.Timestamp(processed_df.at[idx, 'ONSET_DATE'])
                processed_df.at[idx, 'DAYS_TO_REPORT'] = (receive_date - onset_date).days
            except:
                # If any calculation fails, set to 0
                processed_df.at[idx, 'DAYS_TO_REPORT'] = 0

    # Drop original date columns
    for col in date_columns:
        if col in processed_df.columns:
            processed_df = processed_df.drop(col, axis=1)

    # Handle TREATED_FOR_AE - convert to binary (1/0)
    if 'TREATED_FOR_AE' in processed_df.columns:
        # Convert to binary (assuming it's similar to SERIOUS_AE)
        processed_df['TREATED_FOR_AE'] = processed_df['TREATED_FOR_AE'].fillna(0).astype(int)
        print("\nTREATED_FOR_AE value counts:")
        print(processed_df['TREATED_FOR_AE'].value_counts())
    
    # Create age categories
    processed_df['AGE_CATEGORY'] = pd.cut(
        processed_df['ANIMAL_AGE'],
        bins=[0, 1, 3, 7, 100],
        labels=['Puppy', 'Young', 'Adult', 'Senior']
    )
    
    # Create weight categories
    processed_df['WEIGHT_CATEGORY'] = pd.cut(
        processed_df['ANIMAL_WEIGHT'],
        bins=[0, 5, 15, 30, 100],
        labels=['Small', 'Medium', 'Large', 'Extra Large']
    )
    
    # Process reactions column
    if 'REACTIONS' in processed_df.columns:
        processed_df['REACTIONS'] = processed_df['REACTIONS'].fillna('')
        
        # Extract common reactions
        common_reactions = ['vomiting', 'lethargy', 'anorexia', 'seizure', 'diarrhea', 
                        'agitation', 'panting', 'pain', 'pruritus', 'death']
        
        for reaction in common_reactions:
            processed_df[f'HAS_{reaction.upper()}'] = processed_df['REACTIONS'].str.contains(
                reaction, case=False, na=False).astype(int)
        
        # Create reaction count feature
        processed_df['NUM_REACTIONS'] = processed_df['REACTIONS'].apply(
            lambda x: len(x.split(', ')) if x else 0
        )

    # Process breeds - focus on top 10 breeds and group others
    if 'BREEDS' in processed_df.columns:
        processed_df['BREEDS'] = processed_df['BREEDS'].fillna('')
        
        # Define our top breeds list
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
            processed_df[col_name] = 0
        
        # Process each breed entry
        for idx, breeds_str in processed_df['BREEDS'].items():
            if not breeds_str:
                processed_df.at[idx, 'BREED_Other'] = 1
                continue
                
            breeds_list = [b.strip() for b in breeds_str.split(',')]
            found_top_breed = False
            
            for breed in breeds_list:
                # Check if this breed matches any of our top breeds
                for top_breed in top_breeds:
                    if top_breed.lower() in breed.lower():
                        col_name = f'BREED_{top_breed.replace(" ", "_").replace("-", "_")}'
                        processed_df.at[idx, col_name] = 1
                        found_top_breed = True
                        break
                
                # If we found a top breed, we can stop checking other breeds for this record
                if found_top_breed:
                    break
            
            # If no top breed was found, mark as "Other"
            if not found_top_breed:
                processed_df.at[idx, 'BREED_Other'] = 1

    # Handle other categorical columns
    categorical_cols = ['ANIMAL_SPECIES', 'ANIMAL_GENDER', 'ANIMAL_REPRODUCTIVE_STATUS', 
                        'ANIMAL_BREED_IS_CROSSBRED', 'HEALTH_ASSESSMENT_PRIOR_TO_EXPOSURE_CONDITION','MEDICAL_STATUS',
                        'AGE_CATEGORY', 'WEIGHT_CATEGORY']
    
    for col in categorical_cols:
        if col in processed_df.columns:
            # Fill missing values with the most frequent value
            most_frequent = processed_df[col].mode()[0] if not processed_df[col].isna().all() else 'Unknown'
            processed_df[col] = processed_df[col].fillna(most_frequent)
    
    # Handle SERIOUS_AE target variable
    if 'SERIOUS_AE' in processed_df.columns:
        processed_df['SERIOUS_AE'] = processed_df['SERIOUS_AE'].fillna(0).astype(int)
    
    # Drop rows with missing target values
    if 'SERIOUS_AE' in processed_df.columns:
        processed_df = processed_df.dropna(subset=['SERIOUS_AE'])
    
    # Visualize target distribution
    if 'SERIOUS_AE' in processed_df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=processed_df['SERIOUS_AE'])
        plt.title('Distribution of Serious Adverse Events')
        plt.xticks([0, 1], ['False (0)', 'True (1)'])
        plt.tight_layout()
        plt.savefig('serious_ae_distribution.png')
    
    # Visualize relationship between TREATED_FOR_AE and SERIOUS_AE
    if 'TREATED_FOR_AE' in processed_df.columns and 'SERIOUS_AE' in processed_df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='TREATED_FOR_AE', hue='SERIOUS_AE', data=processed_df)
        plt.title('Relationship between TREATED_FOR_AE and SERIOUS_AE')
        plt.xticks([0, 1], ['Not Treated (0)', 'Treated (1)'])
        plt.legend(title='SERIOUS_AE', labels=['No', 'Yes'])
        plt.tight_layout()
        plt.savefig('treated_ae_vs_serious_ae.png')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Age distribution
        sns.histplot(data=processed_df, x='ANIMAL_AGE', kde=True, ax=ax1)
        ax1.set_title('Distribution of Animal Age')
        ax1.set_xlabel('Age (years)')
        ax1.set_ylabel('Count')

        # Weight distribution
        sns.histplot(data=processed_df, x='ANIMAL_WEIGHT', kde=True, ax=ax2)
        ax2.set_title('Distribution of Animal Weight')
        ax2.set_xlabel('Weight (kg)')
        ax2.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig('age_weight_distribution.png')   

        reaction_columns = [col for col in processed_df.columns if col.startswith('HAS_')]
        reaction_counts = processed_df[reaction_columns].sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=reaction_counts.values, y=reaction_counts.index)
        plt.title('Frequency of Different Reaction Types')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig('reaction_types_frequency.png')

        breed_columns = [col for col in processed_df.columns if col.startswith('BREED_')]
        breed_counts = processed_df[breed_columns].sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=breed_counts.values, y=breed_counts.index)
        plt.title('Frequency of Different Dog Breeds')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig('breed_frequency.png')
                
    return processed_df

# Function to prepare data for modeling
def prepare_model_data(df):
    # Make a copy
    model_df = df.copy()
    
    # Identify categorical and numerical columns
    categorical_features = model_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = model_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target variable from features
    if 'SERIOUS_AE' in numeric_features:
        numeric_features.remove('SERIOUS_AE')
    
    # Also remove original text columns as we've created features from them
    if 'REACTIONS' in categorical_features:
        categorical_features.remove('REACTIONS')
    if 'BREEDS' in categorical_features:
        categorical_features.remove('BREEDS')
    
    # Create preprocessor
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # This drops any columns not specified
    )
    
    return model_df, preprocessor, numeric_features, categorical_features

# Function to train and evaluate models for SERIOUS_AE prediction
def train_serious_ae_model(df, preprocessor):
    # Separate features and target
    y = df['SERIOUS_AE']
    X = df.drop(['SERIOUS_AE'], axis=1, errors='ignore')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        start_time = time.time()
        print(f"\nTraining {name} for SERIOUS_AE prediction...")
        
        # Create pipeline with preprocessor and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'training_time': training_time,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'X_test': X_test  # Store X_test for later use
        }
        
        # Print results
        print(f"{name} Results for SERIOUS_AE prediction:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name} (SERIOUS_AE)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'serious_ae_confusion_matrix_{name.replace(" ", "_").lower()}.png')
    
    return results

# Function to visualize model comparison
def visualize_model_comparison(results):
    # Get model names and metrics
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    precisions = [results[model]['precision'] for model in model_names]
    recalls = [results[model]['recall'] for model in model_names]
    f1_scores = [results[model]['f1_score'] for model in model_names]
    roc_aucs = [results[model]['roc_auc'] for model in model_names]
    
    # Create plot
    plt.figure(figsize=(15, 12))
    
    # Plot accuracy
    plt.subplot(3, 2, 1)
    sns.barplot(x=model_names, y=accuracies)
    plt.title('SERIOUS_AE - Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Plot precision
    plt.subplot(3, 2, 2)
    sns.barplot(x=model_names, y=precisions)
    plt.title('SERIOUS_AE - Model Precision Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Plot recall
    plt.subplot(3, 2, 3)
    sns.barplot(x=model_names, y=recalls)
    plt.title('SERIOUS_AE - Model Recall Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Plot F1 score
    plt.subplot(3, 2, 4)
    sns.barplot(x=model_names, y=f1_scores)
    plt.title('SERIOUS_AE - Model F1 Score Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Plot ROC AUC
    plt.subplot(3, 2, 5)
    sns.barplot(x=model_names, y=roc_aucs)
    plt.title('SERIOUS_AE - Model ROC AUC Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    plt.savefig('serious_ae_model_comparison.png')
    
    # Create dataframe for results
    results_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'ROC AUC': roc_aucs
    })
    
    return results_df

# Function to analyze feature importance for best model
def analyze_feature_importance(best_model, numeric_features, categorical_features, preprocessor):
    if hasattr(best_model, 'feature_importances_'):
        # Get feature names from preprocessor
        feature_names = []
        
        # Get numerical feature names directly
        for feature in numeric_features:
            feature_names.append(feature)
        
        # Get one-hot encoded categorical feature names
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_feature_names = []
        for i, category in enumerate(ohe.categories_):
            for cat_value in category:
                categorical_feature_names.append(f"{categorical_features[i]}_{cat_value}")
        
        feature_names.extend(categorical_feature_names)
        
        # Get feature importances
        importances = best_model.feature_importances_
        
        # Create dataframe with feature importances
        feature_importances = pd.DataFrame({
            'Feature': feature_names[:len(importances)],  # In case there's a mismatch
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
        plt.title('Top 20 Feature Importances for SERIOUS_AE Prediction')
        plt.tight_layout()
        plt.savefig('serious_ae_feature_importance.png')
        
        return feature_importances
    else:
        print("Selected model doesn't support feature importances")
        return None

# New function to analyze and save actual vs predicted values
def analyze_predictions(model_name, y_test, y_pred, y_prob, X_test):
    print(f"\nAnalyzing predictions for {model_name}:")
    
    # Create DataFrame with actual and predicted values
    predictions_df = pd.DataFrame({
        'Actual_SERIOUS_AE': y_test.values,
        'Predicted_SERIOUS_AE': y_pred,
        'Probability_SERIOUS_AE': y_prob if y_prob is not None else np.nan
    })
    
    # Add selected features from X_test if needed
    # This will include original index values
    predictions_df.index = X_test.index
    
    # Display the first few rows
    print("\nSample of Actual vs Predicted values:")
    print(predictions_df.head(10))
    
    # Count instances by prediction category
    print("\nCounts by Prediction Category:")
    prediction_counts = pd.crosstab(
        predictions_df['Actual_SERIOUS_AE'], 
        predictions_df['Predicted_SERIOUS_AE'],
        rownames=['Actual'],
        colnames=['Predicted']
    )
    print(prediction_counts)
    
    # Save predictions to CSV
    predictions_file = f'serious_ae_predictions_{model_name.replace(" ", "_").lower()}.csv'
    predictions_df.to_csv(predictions_file)
    print(f"\nPredictions saved to {predictions_file}")
    
    # Create ROC curve plot
    if y_prob is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_prob):.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.savefig(f'serious_ae_roc_curve_{model_name.replace(" ", "_").lower()}.png')
    
    return predictions_df

# Main execution function
def main():
    # Connect to Snowflake and fetch data
    conn = connect_to_snowflake(snowflake_params)
    
    if conn:
        dog_health_df = fetch_dog_health_data(conn)
        conn.close()  # Close the connection when done
        
        if dog_health_df is not None:
            # Display basic information about the dataset
            print("\nDataset Information:")
            print(f"Shape: {dog_health_df.shape}")
            print("\nColumn Names:")
            print(dog_health_df.columns.tolist())
            print("\nData Types:")
            print(dog_health_df.dtypes)
            print("\nSample Data:")
            print(dog_health_df.head())
            
            # Preprocess data
            processed_df = preprocess_data(dog_health_df)
            
            if processed_df is not None:
                # Prepare data for modeling
                model_df, preprocessor, numeric_features, categorical_features = prepare_model_data(processed_df)
                
                # Train and evaluate models for SERIOUS_AE prediction
                serious_ae_results = train_serious_ae_model(model_df, preprocessor)
                
                # Compare model results
                results_df = visualize_model_comparison(serious_ae_results)
                
                print("\nSERIOUS_AE Model Comparison:")
                print(results_df.sort_values('F1 Score', ascending=False))
                
                # Get best model
                best_model_name = results_df.sort_values('F1 Score', ascending=False).iloc[0]['Model']
                best_model = serious_ae_results[best_model_name]['pipeline'].named_steps['classifier']
                
                print(f"\nBest model for SERIOUS_AE prediction: {best_model_name}")
                
                # Analyze feature importance for best model if it's a tree-based model
                if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                    feature_importances = analyze_feature_importance(
                        best_model, numeric_features, categorical_features, 
                        serious_ae_results[best_model_name]['pipeline'].named_steps['preprocessor']
                    )
                    
                    if feature_importances is not None:
                        print("\nTop 10 Features for SERIOUS_AE Prediction:")
                        print(feature_importances.head(10))
                
                # Save best model
                joblib.dump(serious_ae_results[best_model_name]['pipeline'], 'best_serious_ae_model.pkl')
                
                print("\nBest model saved successfully!")
                
                # NEW SECTION: Get and analyze actual vs predicted values for the best model
                best_y_test = serious_ae_results[best_model_name]['y_test']
                best_y_pred = serious_ae_results[best_model_name]['y_pred']
                best_y_prob = serious_ae_results[best_model_name]['y_prob']
                best_X_test = serious_ae_results[best_model_name]['X_test']
                
                # Analyze predictions from the best model
                predictions_df = analyze_predictions(best_model_name, best_y_test, best_y_pred, best_y_prob, best_X_test)
                
                # Find the most challenging misclassifications
                if len(predictions_df) > 0:
                    # False negatives (actual=1, predicted=0) - missed serious AEs
                    false_negatives = predictions_df[(predictions_df['Actual_SERIOUS_AE'] == 1) & 
                                                     (predictions_df['Predicted_SERIOUS_AE'] == 0)]
                    if len(false_negatives) > 0:
                        print("\nSample of False Negatives (Missed Serious AEs):")
                        print(false_negatives.head(5))
                        false_negatives.to_csv(f'serious_ae_false_negatives_{best_model_name.replace(" ", "_").lower()}.csv')
                    
                    # False positives (actual=0, predicted=1) - incorrectly flagged as serious
                    false_positives = predictions_df[(predictions_df['Actual_SERIOUS_AE'] == 0) & 
                                                     (predictions_df['Predicted_SERIOUS_AE'] == 1)]
                    if len(false_positives) > 0:
                        print("\nSample of False Positives (Incorrectly Flagged as Serious):")
                        print(false_positives.head(5))
                        false_positives.to_csv(f'serious_ae_false_positives_{best_model_name.replace(" ", "_").lower()}.csv')
                
                print("\nAnalysis complete!")
            else:
                print("Error: Failed to preprocess data")
        else:
            print("Error: Failed to fetch data from Snowflake")
    else:
        print("Error: Failed to connect to Snowflake")

if __name__ == "__main__":
    main()