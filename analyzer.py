# UFC Fight Outcome Prediction Script (Enhanced with Class Imbalance Handling)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from collections import Counter

import joblib
import warnings

warnings.filterwarnings('ignore')


def load_data(fight_data_path='ufc_fight_data.csv', fight_stats_path='ufc_fight_stat_data.csv'):
    try:
        ufc_fights = pd.read_csv(fight_data_path)
        print("UFC Fights Data Loaded Successfully.")
        print(ufc_fights.head())
    except FileNotFoundError:
        print(f"Error: '{fight_data_path}' not found.")
        exit()

    try:
        ufc_fight_stats = pd.read_csv(fight_stats_path)
        print("\nUFC Fight Stats Data Loaded Successfully.")
        print(ufc_fight_stats.head())
    except FileNotFoundError:
        print(f"Error: '{fight_stats_path}' not found.")
        exit()

    return ufc_fights, ufc_fight_stats


def preprocess_data(ufc_fights, ufc_fight_stats):
    # Handle missing values
    ufc_fights.dropna(subset=['fight_id'], inplace=True)
    ufc_fight_stats.dropna(subset=['fight_id', 'fighter_id'], inplace=True)

    # Convert 'ctrl_time' to seconds
    def convert_ctrl_time(time_str):
        if pd.isnull(time_str):
            return np.nan
        if time_str == '--':
            return 0
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except:
            return np.nan

    ufc_fight_stats['ctrl_time_seconds'] = ufc_fight_stats['ctrl_time'].apply(convert_ctrl_time)
    median_ctrl_time = ufc_fight_stats['ctrl_time_seconds'].median()
    ufc_fight_stats['ctrl_time_seconds'].fillna(median_ctrl_time, inplace=True)
    ufc_fight_stats.drop('ctrl_time', axis=1, inplace=True)

    # Convert 'finish_time' to seconds
    def convert_finish_time(time_str):
        if pd.isnull(time_str):
            return np.nan
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds
        except:
            return np.nan

    ufc_fights['finish_time_seconds'] = ufc_fights['finish_time'].apply(convert_finish_time)
    ufc_fights.dropna(subset=['finish_time_seconds'], inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    ufc_fights['weight_class'].fillna('Unknown', inplace=True)
    ufc_fights['weight_class_encoded'] = le.fit_transform(ufc_fights['weight_class'])
    ufc_fights['gender_encoded'] = le.fit_transform(ufc_fights['gender'])

    # Encode 'title_fight'
    title_fight_mapping = {'F': 0, 'T': 1}
    ufc_fights['title_fight'] = ufc_fights['title_fight'].map(title_fight_mapping)
    ufc_fights['title_fight'].fillna(0, inplace=True)

    return ufc_fights, ufc_fight_stats


def merge_and_engineer_features(ufc_fights, ufc_fight_stats):
    # Merge datasets
    merged_stats = pd.merge(ufc_fight_stats, ufc_fights[['fight_id', 'f_1', 'f_2']], on='fight_id', how='left')

    # Assign fighter roles
    merged_stats['fighter_role'] = np.where(merged_stats['fighter_id'] == merged_stats['f_1'], 'Fighter1',
                                            np.where(merged_stats['fighter_id'] == merged_stats['f_2'], 'Fighter2',
                                                     'Other'))
    merged_stats = merged_stats[merged_stats['fighter_role'] != 'Other']

    # Pivot for Fighter1 and Fighter2
    fighter1_stats = merged_stats[merged_stats['fighter_role'] == 'Fighter1'].copy()
    fighter2_stats = merged_stats[merged_stats['fighter_role'] == 'Fighter2'].copy()

    columns_to_rename_f1 = {col: f"{col}_F1" for col in fighter1_stats.columns if
                            col not in ['fight_id', 'fighter_id', 'fighter_role', 'fight_url']}
    columns_to_rename_f2 = {col: f"{col}_F2" for col in fighter2_stats.columns if
                            col not in ['fight_id', 'fighter_id', 'fighter_role', 'fight_url']}

    fighter1_stats.rename(columns=columns_to_rename_f1, inplace=True)
    fighter2_stats.rename(columns=columns_to_rename_f2, inplace=True)

    fight_data = pd.merge(fighter1_stats, fighter2_stats, on='fight_id', how='inner')
    print("\n--- Merged Fight Data ---")
    print(fight_data.head())

    # Feature Engineering
    fight_data['total_strikes_diff'] = fight_data['total_strikes_succ_F1'] - fight_data['total_strikes_succ_F2']
    fight_data['knockdowns_diff'] = fight_data['knockdowns_F1'] - fight_data['knockdowns_F2']
    fight_data['takedowns_diff'] = fight_data['takedown_succ_F1'] - fight_data['takedown_succ_F2']
    fight_data['ctrl_time_diff'] = fight_data['ctrl_time_seconds_F1'] - fight_data['ctrl_time_seconds_F2']

    fight_data['strike_accuracy_F1'] = fight_data['sig_strikes_succ_F1'] / (fight_data['sig_strikes_att_F1'] + 1)
    fight_data['strike_accuracy_F2'] = fight_data['sig_strikes_succ_F2'] / (fight_data['sig_strikes_att_F2'] + 1)
    fight_data['strike_accuracy_diff'] = fight_data['strike_accuracy_F1'] - fight_data['strike_accuracy_F2']

    fight_data['submission_ratio_F1'] = fight_data['submission_att_F1'] / (fight_data['takedown_att_F1'] + 1)
    fight_data['submission_ratio_F2'] = fight_data['submission_att_F2'] / (fight_data['takedown_att_F2'] + 1)
    fight_data['submission_ratio_diff'] = fight_data['submission_ratio_F1'] - fight_data['submission_ratio_F2']

    # Merge weight_class_encoded and gender_encoded
    fight_data = pd.merge(fight_data, ufc_fights[['fight_id', 'weight_class_encoded', 'gender_encoded', 'title_fight']],
                          on='fight_id', how='left')

    # Prepare target variable
    winner_mapping = ufc_fights.set_index('fight_id')['winner'].to_dict()
    fight_data['winner_fighter_id'] = fight_data['fight_id'].map(winner_mapping)
    fight_data['fight_winner'] = np.where(fight_data['f_1_F1'] == fight_data['winner_fighter_id'], 0,
                                          np.where(fight_data['f_2_F2'] == fight_data['winner_fighter_id'], 1, np.nan))
    fight_data.dropna(subset=['fight_winner'], inplace=True)
    fight_data['fight_winner'] = fight_data['fight_winner'].astype(int)

    print("\n--- Target Variable Distribution ---")
    print(fight_data['fight_winner'].value_counts())

    return fight_data


def create_symmetric_dataset(fight_data):
    # Create a copy with F1 and F2 swapped
    fight_data_swapped = fight_data.copy()
    fight_data_swapped['total_strikes_diff'] = -fight_data_swapped['total_strikes_diff']
    fight_data_swapped['knockdowns_diff'] = -fight_data_swapped['knockdowns_diff']
    fight_data_swapped['takedowns_diff'] = -fight_data_swapped['takedowns_diff']
    fight_data_swapped['ctrl_time_diff'] = -fight_data_swapped['ctrl_time_diff']
    fight_data_swapped['strike_accuracy_diff'] = -fight_data_swapped['strike_accuracy_diff']
    fight_data_swapped['submission_ratio_diff'] = -fight_data_swapped['submission_ratio_diff']

    # Invert the target variable
    fight_data_swapped['fight_winner'] = 1 - fight_data_swapped['fight_winner']

    # Append the swapped data to the original dataset
    symmetric_fight_data = pd.concat([fight_data, fight_data_swapped], ignore_index=True)

    print("\n--- Symmetric Dataset Created ---")
    print(symmetric_fight_data['fight_winner'].value_counts())

    return symmetric_fight_data


def select_features_and_split(fight_data):
    feature_cols = [
        'total_strikes_diff',
        'knockdowns_diff',
        'takedowns_diff',
        'ctrl_time_diff',
        'strike_accuracy_diff',
        'submission_ratio_diff',
        'weight_class_encoded',
        'gender_encoded',
        'title_fight'
    ]

    X = fight_data[feature_cols]
    y = fight_data['fight_winner']

    print("\n--- Missing Values in Features ---")
    print(X.isnull().sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining Set Size: {X_train.shape}")
    print(f"Testing Set Size: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols


def handle_imputation_and_scaling(X_train, X_test, feature_cols):
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Verify no missing values post-imputation
    print("\n--- Post-Imputation Missing Values in Training Set ---")
    print(pd.DataFrame(X_train_imputed, columns=feature_cols).isnull().sum())

    print("\n--- Post-Imputation Missing Values in Testing Set ---")
    print(pd.DataFrame(X_test_imputed, columns=feature_cols).isnull().sum())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, imputer, scaler


def handle_class_imbalance(X_train, y_train):
    print("\n--- Before Resampling ---")
    print(Counter(y_train))

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("\n--- After Resampling ---")
    print(Counter(y_res))

    return X_res, y_res


def train_models(X_train, y_train, X_test, y_test):
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    print("\n--- Logistic Regression Classification Report ---")
    print(classification_report(y_test, y_pred_lr))

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_test, y_pred_rf))

    # Gradient Boosting
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(X_train, y_train)
    y_pred_gb = gb_clf.predict(X_test)
    print("\n--- Gradient Boosting Classification Report ---")
    print(classification_report(y_test, y_pred_gb))

    return log_reg, rf_clf, gb_clf


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fighter1', 'Fighter2'],
                yticklabels=['Fighter1', 'Fighter2'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()


def hyperparameter_tuning(X_train, y_train):
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    random_search_rf = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_distributions=param_grid_rf,
        n_iter=24,  # Number of parameter settings sampled
        cv=5,
        verbose=2,
        random_state=42,
        scoring='accuracy',
        n_jobs=-1
    )

    print("\nStarting Randomized Search for Random Forest...")
    random_search_rf.fit(X_train, y_train)

    print(f"\nBest Parameters for Random Forest: {random_search_rf.best_params_}")

    best_rf = random_search_rf.best_estimator_

    return best_rf


def cross_validate_model(model, X_train, y_train):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")


def feature_importance_plot(model, feature_names):
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importances - Tuned Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()


def save_model(model, scaler, imputer, model_path='ufc_fight_outcome_model.pkl', scaler_path='scaler.pkl',
               imputer_path='imputer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    print(f"\nTrained Random Forest model saved as '{model_path}'.")
    print(f"Scaler saved as '{scaler_path}'.")
    print(f"Imputer saved as '{imputer_path}'.")


def load_model(model_path='ufc_fight_outcome_model.pkl', scaler_path='scaler.pkl', imputer_path='imputer.pkl'):
    try:
        loaded_model = joblib.load(model_path)
        print("\nModel loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{model_path}' not found.")
        exit()

    try:
        loaded_scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{scaler_path}' not found.")
        exit()

    try:
        loaded_imputer = joblib.load(imputer_path)
        print("Imputer loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{imputer_path}' not found.")
        exit()

    return loaded_model, loaded_scaler, loaded_imputer


def predict_fight_outcome(model, imputer, scaler, fight_stats, feature_cols):
    fight_df = pd.DataFrame([fight_stats])

    # Feature Engineering
    fight_df['total_strikes_diff'] = fight_df['total_strikes_succ_F1'] - fight_df['total_strikes_succ_F2']
    fight_df['knockdowns_diff'] = fight_df['knockdowns_F1'] - fight_df['knockdowns_F2']
    fight_df['takedowns_diff'] = fight_df['takedown_succ_F1'] - fight_df['takedown_succ_F2']
    fight_df['ctrl_time_diff'] = fight_df['ctrl_time_seconds_F1'] - fight_df['ctrl_time_seconds_F2']

    fight_df['strike_accuracy_diff'] = (fight_df['sig_strikes_succ_F1'] / (fight_df['sig_strikes_att_F1'] + 1)) - \
                                       (fight_df['sig_strikes_succ_F2'] / (fight_df['sig_strikes_att_F2'] + 1))

    fight_df['submission_ratio_diff'] = (fight_df['submission_att_F1'] / (fight_df['takedown_att_F1'] + 1)) - \
                                        (fight_df['submission_att_F2'] / (fight_df['takedown_att_F2'] + 1))

    # Prepare feature set
    feature_set = fight_df[feature_cols]

    # Handle missing values
    feature_set_imputed = imputer.transform(feature_set)

    # Scale features
    feature_set_scaled = scaler.transform(feature_set_imputed)

    # Make prediction
    prediction = model.predict(feature_set_scaled)[0]

    return 'Fighter2' if prediction == 1 else 'Fighter1'


def main():
    # Load Data
    ufc_fights, ufc_fight_stats = load_data()

    # Preprocess Data
    ufc_fights, ufc_fight_stats = preprocess_data(ufc_fights, ufc_fight_stats)

    # Merge and Engineer Features
    fight_data = merge_and_engineer_features(ufc_fights, ufc_fight_stats)

    # Create Symmetric Dataset
    symmetric_fight_data = create_symmetric_dataset(fight_data)

    # Select Features and Split
    X_train, X_test, y_train, y_test, feature_cols = select_features_and_split(symmetric_fight_data)

    # Handle Imputation and Scaling
    X_train_scaled, X_test_scaled, imputer, scaler = handle_imputation_and_scaling(X_train, X_test, feature_cols)

    # Handle Class Imbalance with SMOTE
    X_train_res, y_train_res = handle_class_imbalance(X_train_scaled, y_train)

    # Train Models on Resampled Data
    log_reg, rf_clf, gb_clf = train_models(X_train_res, y_train_res, X_test_scaled, y_test)

    # Plot Confusion Matrices
    y_pred_lr = log_reg.predict(X_test_scaled)
    plot_confusion(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")

    y_pred_rf = rf_clf.predict(X_test_scaled)
    plot_confusion(y_test, y_pred_rf, "Random Forest Confusion Matrix")

    y_pred_gb = gb_clf.predict(X_test_scaled)
    plot_confusion(y_test, y_pred_gb, "Gradient Boosting Confusion Matrix")

    # Hyperparameter Tuning
    best_rf = hyperparameter_tuning(X_train_res, y_train_res)

    # Evaluate Tuned Model
    y_pred_best_rf = best_rf.predict(X_test_scaled)
    print("\n--- Tuned Random Forest Classification Report ---")
    print(classification_report(y_test, y_pred_best_rf))

    plot_confusion(y_test, y_pred_best_rf, "Tuned Random Forest Confusion Matrix")

    # Cross-Validation
    cross_validate_model(best_rf, X_train_res, y_train_res)

    # Feature Importance
    feature_importance_plot(best_rf, feature_cols)

    # Save Model, Scaler, and Imputer
    save_model(best_rf, scaler, imputer)

    # Load Model, Scaler, and Imputer
    loaded_model, loaded_scaler, loaded_imputer = load_model()

    # Example Prediction
    sample_fight = {
        'total_strikes_succ_F1': 100,  # +1 difference for Fighter 1
        'total_strikes_succ_F2': 101,
        'knockdowns_F1': 2,
        'knockdowns_F2': 2,
        'takedown_succ_F1': 3,
        'takedown_succ_F2': 3,
        'ctrl_time_seconds_F1': 300,  # in seconds
        'ctrl_time_seconds_F2': 300,
        'sig_strikes_succ_F1': 80,
        'sig_strikes_att_F1': 120,
        'sig_strikes_succ_F2': 80,
        'sig_strikes_att_F2': 120,
        'submission_att_F1': 4,
        'submission_att_F2': 4,
        'takedown_att_F1': 6,
        'takedown_att_F2': 6,
        'weight_class_encoded': 3,  # Same value
        'gender_encoded': 1,  # Same value
        'title_fight': 0  # Same value
    }

    predicted_winner = predict_fight_outcome(loaded_model, loaded_imputer, loaded_scaler, sample_fight, feature_cols)
    print(f"\nPredicted Winner: {predicted_winner}")

    # Additional Best Practices
    # Handling Class Imbalance
    print("\n--- Class Distribution ---")
    print(fight_data['fight_winner'].value_counts())

    # Balanced Random Forest (Alternative Approach)
    rf_clf_balanced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_clf_balanced.fit(X_train_res, y_train_res)
    y_pred_rf_balanced = rf_clf_balanced.predict(X_test_scaled)

    print("\n--- Balanced Random Forest Classification Report ---")
    print(classification_report(y_test, y_pred_rf_balanced))

    # Ensemble Methods - Voting Classifier
    log_reg_ensemble = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    gb_clf_ensemble = GradientBoostingClassifier(n_estimators=100, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('rf', best_rf), ('lr', log_reg_ensemble), ('gb', gb_clf_ensemble)],
        voting='hard',
        n_jobs=-1
    )

    voting_clf.fit(X_train_res, y_train_res)
    y_pred_voting = voting_clf.predict(X_test_scaled)

    print("\n--- Voting Classifier Classification Report ---")
    print(classification_report(y_test, y_pred_voting))

    plot_confusion(y_test, y_pred_voting, "Voting Classifier Confusion Matrix")




if __name__ == "__main__":
    main()
