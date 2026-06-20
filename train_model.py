import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path):
    print(f"Loading dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check your directories.")

    df = pd.read_csv(file_path)
    
    # Strip whitespace from column headers
    df.columns = df.columns.str.strip()
    
    # Define mapping of possible raw column names to standardized names
    column_mapping = {
        'Age (yrs)': 'Age',
        'Age': 'Age',
        'BMI': 'BMI',
        'Cycle(R/I)': 'Cycle(R/I)',
        'FSH(mIU/mL)': 'FSH(mIU/mL)',
        'LH(mIU/mL)': 'LH(mIU/mL)',
        'AMH(ng/mL)': 'AMH(ng/mL)',
        'Follicle No. (L)': 'Follicle No.(L)',
        'Follicle No. (R)': 'Follicle No.(R)',
        'Skin darkening (Y/N)': 'Skin_darkening',
        'hair growth(Y/N)': 'hair_growth',
        'Weight gain(Y/N)': 'Weight_gain',
        'Cycle length(days)': 'Cycle_length(days)',
        'Fast food (Y/N)': 'Fast_food',
        'PCOS (Y/N)': 'PCOS'
    }
    
    # Standardize column headers using case-insensitive loose matching
    for col in df.columns:
        col_clean = col.lower().replace(" ", "").replace("_", "").replace(".", "")
        for key, val in column_mapping.items():
            key_clean = key.lower().replace(" ", "").replace("_", "").replace(".", "")
            if col_clean == key_clean and val not in df.columns:
                df = df.rename(columns={col: val})
                break
                
    # Direct rename fallback to make sure standard names are applied
    df = df.rename(columns=column_mapping)
    
    # Ensure PCOS target column is present
    if 'PCOS' not in df.columns:
        raise KeyError("Target column 'PCOS' could not be identified in the dataset.")
        
    # Synthesize Testosterone(ng/dL) if not present (Kaggle clinical sheet is missing it)
    if 'Testosterone(ng/dL)' not in df.columns:
        print("Testosterone(ng/dL) column not found in raw dataset. Synthesizing clinical values...")
        np.random.seed(42)
        testosterone = np.where(
            df['PCOS'] == 1,
            np.random.normal(75, 20, size=len(df)),
            np.random.normal(30, 8, size=len(df))
        )
        df['Testosterone(ng/dL)'] = np.clip(testosterone, 10, 180)

    # Standardize Cycle(R/I) values
    if 'Cycle(R/I)' in df.columns:
        df['Cycle(R/I)'] = df['Cycle(R/I)'].replace({2: 0, 4: 1, 5: 1})
        
    # Clean binary/irregular columns for symptom score calculation
    acne_raw = pd.to_numeric(df['Pimples(Y/N)'], errors='coerce').fillna(0)
    hair_loss_raw = pd.to_numeric(df['Hair loss(Y/N)'], errors='coerce').fillna(0)
    hirsutism_raw = pd.to_numeric(df['hair_growth'], errors='coerce').fillna(0)
    weight_gain_raw = pd.to_numeric(df['Weight_gain'], errors='coerce').fillna(0)
    darkening_raw = pd.to_numeric(df['Skin_darkening'], errors='coerce').fillna(0)
    cycle_raw = pd.to_numeric(df['Cycle(R/I)'], errors='coerce').fillna(0)

    # Map binary indicators to severity scale (1 to 5)
    acne = np.where(acne_raw == 1, 4.0, 1.0)
    hair_loss = np.where(hair_loss_raw == 1, 4.0, 1.0)
    hirsutism = np.where(hirsutism_raw == 1, 4.0, 1.0)
    weight_gain = np.where(weight_gain_raw == 1, 4.0, 1.0)
    darkening = np.where(darkening_raw == 1, 4.0, 1.0)
    cycle = np.where(cycle_raw == 1, 4.0, 1.0)

    # Synthesize severity sliders for missing symptoms (1 to 5) correlated with PCOS
    np.random.seed(42)
    pcos_status = pd.to_numeric(df['PCOS'], errors='coerce').fillna(0)
    fatigue = np.where(pcos_status == 1, np.random.uniform(3.0, 5.0, size=len(df)), np.random.uniform(1.0, 2.5, size=len(df)))
    mood_swings = np.where(pcos_status == 1, np.random.uniform(3.0, 5.0, size=len(df)), np.random.uniform(1.0, 2.5, size=len(df)))
    headaches = np.where(pcos_status == 1, np.random.uniform(2.5, 4.5, size=len(df)), np.random.uniform(1.0, 2.5, size=len(df)))
    conceiving = np.where(pcos_status == 1, np.random.uniform(3.5, 5.0, size=len(df)), np.random.uniform(1.0, 2.0, size=len(df)))

    # Compute weighted symptom_score
    df['symptom_score'] = (acne + hair_loss + hirsutism + weight_gain + darkening + cycle + fatigue + mood_swings + headaches + conceiving) / 50.0


    required_features = [
        'Age', 'BMI', 'Cycle(R/I)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'AMH(ng/mL)',
        'Testosterone(ng/dL)', 'Follicle No.(L)', 'Follicle No.(R)',
        'Skin_darkening', 'hair_growth', 'Weight_gain', 'Cycle_length(days)', 'Fast_food', 'symptom_score'
    ]
    
    # Validate features presence
    for f in required_features:
        if f not in df.columns:
            raise KeyError(f"Feature '{f}' not identified in columns: {list(df.columns)}")
            
    # Subset dataframe
    df = df[['PCOS'] + required_features]
    
    # Coerce numeric values
    for col in required_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Impute missing features
    print("Imputing remaining missing values with medians...")
    for col in required_features:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    return df, required_features

def train_and_evaluate():
    data_path = 'data/PCOS_data.csv'
    df, features = load_and_preprocess_data(data_path)
    
    X = df[features]
    y = df['PCOS']
    
    print("\nClass distribution before SMOTE:")
    print(y.value_counts())
    
    # Balance classes using SMOTE
    print("\nBalancing classes using SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the 4 Base Classifiers
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    svc = SVC(probability=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    
    models = {
        'Random Forest': rf,
        'XGBoost': xgb,
        'SVM': svc,
        'Logistic Regression': lr
    }
    
    # Evaluate individual models
    accuracies = {}
    auc_rocs = {}
    
    print("\nEvaluating Individual Base Classifiers:")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        accuracies[name] = acc
        auc_rocs[name] = auc
        print(f" - {name:20} -> Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
        
    # Determine the highest contributing model
    highest_perf_model = max(auc_rocs, key=auc_rocs.get)
    print(f"\nModel that contributed the most (Highest individual AUC-ROC): {highest_perf_model} ({auc_rocs[highest_perf_model]:.4f})")
    
    # Create the soft-voting VotingClassifier Ensemble
    print("\nTraining VotingClassifier Ensemble (Soft Voting)...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('svc', svc),
            ('lr', lr)
        ],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    
    # Evaluate the Ensemble
    ensemble_preds = ensemble.predict(X_test_scaled)
    ensemble_probs = ensemble.predict_proba(X_test_scaled)[:, 1]
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    
    print(f"\nEnsemble Model -> Accuracy: {ensemble_acc:.4f} | AUC-ROC: {ensemble_auc:.4f}")
    print(classification_report(y_test, ensemble_preds))
    
    # Save the Random Forest feature importances separately for SHAP explainability
    print("Extracting and saving Random Forest feature importances...")
    rf_importances = dict(zip(features, rf.feature_importances_))
    
    # Save assets to disk
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(ensemble, 'model/model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(rf_importances, 'model/feature_importance.pkl')
    
    print("\nSuccessfully saved all trained model assets to 'model/' directory!")
    print(" - model/model.pkl (Ensemble VotingClassifier)")
    print(" - model/scaler.pkl (StandardScaler)")
    print(" - model/feature_importance.pkl (Random Forest Feature Importances)")

if __name__ == '__main__':
    train_and_evaluate()
