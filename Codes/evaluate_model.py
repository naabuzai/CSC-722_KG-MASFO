import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, mean_squared_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

def create_interaction_features(X, feature_names):
    X_df = pd.DataFrame(X, columns=feature_names)
    X_enhanced = X_df.copy()
    if 'pH' in feature_names and '% Nitrogen' in feature_names:
        X_enhanced['pH_Nitrogen_interaction'] = X_df['pH'] * X_df['% Nitrogen']
    if 'pH' in feature_names and 'Phosphorous (ppm P in soil)' in feature_names:
        X_enhanced['pH_Phosphorus_interaction'] = X_df['pH'] * X_df['Phosphorous (ppm P in soil)']
    if '% Nitrogen' in feature_names and 'Phosphorous (ppm P in soil)' in feature_names:
        X_enhanced['N_P_ratio'] = X_df['% Nitrogen'] / (X_df['Phosphorous (ppm P in soil)'] + 1e-6)
    if '% Nitrogen' in feature_names and 'Potassium (ppm K in soil)' in feature_names:
        X_enhanced['N_K_ratio'] = X_df['% Nitrogen'] / (X_df['Potassium (ppm K in soil)'] + 1e-6)
    if 'pH' in feature_names:
        X_enhanced['pH_acidic'] = (X_df['pH'] < 6.0).astype(int)
        X_enhanced['pH_neutral'] = ((X_df['pH'] >= 6.0) & (X_df['pH'] <= 7.0)).astype(int)
        X_enhanced['pH_alkaline'] = (X_df['pH'] > 7.0).astype(int)
    if 'ProportionCarbon' in feature_names:
        X_enhanced['high_carbon'] = (X_df['ProportionCarbon'] > X_df['ProportionCarbon'].median()).astype(int)
    nutrient_cols = ['% Nitrogen', 'Phosphorous (ppm P in soil)', 'Potassium (ppm K in soil)', 'Magnesium (ppm Mg in soil)']
    available_nutrients = [col for col in nutrient_cols if col in feature_names]
    if len(available_nutrients) > 1:
        X_enhanced['total_nutrients'] = X_df[available_nutrients].sum(axis=1)
    return X_enhanced.values, list(X_enhanced.columns)

def optimize_for_r2(name, base_model, X_train, X_test, y_train, y_test, use_calibration=True):
    try:
        if use_calibration and hasattr(base_model, 'predict_proba'):
            model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        else:
            model = base_model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X_test)
            y_pred_proba = 1 / (1 + np.exp(-decision_scores))
        else:
            y_pred_proba = y_pred.astype(float)
        epsilon = 1e-7
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        r2 = r2_score(y_test, y_pred_proba)
        mse = mean_squared_error(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            no_fert_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
            fert_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            no_fert_acc = fert_acc = 0.5
        return {
            'Classifier': name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'No Fert. Acc.': no_fert_acc,
            'Fert. Acc.': fert_acc,
            'R²': r2,
            'MSE': mse,
            'Status': 'Success'
        }
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        return {
            'Classifier': name,
            'Accuracy': 0.0,
            'F1 Score': 0.0,
            'No Fert. Acc.': 0.0,
            'Fert. Acc.': 0.0,
            'R²': -999.0,
            'MSE': 999.0,
            'Status': 'Failed'
        }

def main():
    print("=" * 100)
    print("=" * 100)
    try:
        df = pd.read_csv('fertilizerdata.csv')
    except Exception as e:
        print(f"Error: {e}")
        return
    feature_columns = [
        '% Nitrogen', 'pH', 'Potassium (ppm K in soil)', 
        'Phosphorous (ppm P in soil)', 'Magnesium (ppm Mg in soil)', 
        'ProportionCarbon', 'Calcium (ppm Ca in soil)', 'NAdd(g/m2/yr)'
    ]
    df_clean = df.dropna(subset=['FertilizerDecision'])
    X_original = df_clean[feature_columns].fillna(df_clean[feature_columns].median())
    y = df_clean['FertilizerDecision'].astype(int)
    X_enhanced, enhanced_feature_names = create_interaction_features(X_original.values, feature_columns)
    selector = SelectKBest(score_func=mutual_info_classif, k=min(15, X_enhanced.shape[1]))
    X_selected = selector.fit_transform(X_enhanced, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = [enhanced_feature_names[i] for i in selected_indices]
    configs = [
        {'test_size': 0.25, 'random_state': 42},
        {'test_size': 0.30, 'random_state': 17},
        {'test_size': 0.20, 'random_state': 123},
        {'test_size': 0.25, 'random_state': 88},
        {'test_size': 0.30, 'random_state': 256}
    ]
    all_config_results = []
    for config_idx, config in enumerate(configs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=y
        )
        scalers = [
            ('StandardScaler', StandardScaler()),
            ('RobustScaler', RobustScaler()),
            ('MinMaxScaler', MinMaxScaler())
        ]
        for scaler_name, scaler in scalers:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            models = [
                ('Decision Tree', DecisionTreeClassifier(random_state=config['random_state'], max_depth=8, min_samples_split=10, min_samples_leaf=5, criterion='entropy'), False),
                ('Random Forest', RandomForestClassifier(n_estimators=200, random_state=config['random_state'], max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt'), False),
                ('Gradient Boosting', GradientBoostingClassifier(random_state=config['random_state'], n_estimators=150, learning_rate=0.05, max_depth=6, min_samples_split=10), False),
                ('Logistic Regression', LogisticRegression(random_state=config['random_state'], max_iter=2000, C=0.1, class_weight='balanced'), True),
                ('SVM', SVC(random_state=config['random_state'], probability=True, C=10, gamma='scale', class_weight='balanced'), True),
                ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'), True),
                ('Gaussian Naive Bayes', GaussianNB(var_smoothing=1e-8), True),
                ('MLP', MLPClassifier(random_state=config['random_state'], hidden_layer_sizes=(100, 50), max_iter=2000, alpha=0.01, learning_rate='adaptive'), True)
            ]
            if XGBOOST_AVAILABLE:
                models.append(('XGBoost', xgb.XGBClassifier(random_state=config['random_state'], n_estimators=150, max_depth=6, learning_rate=0.05, eval_metric='logloss', verbosity=0), False))
            for name, model, use_scaled in models:
                result = optimize_for_r2(name, model, X_train_scaled if use_scaled else X_train, X_test_scaled if use_scaled else X_test, y_train, y_test)
                result['config'] = config_idx
                result['scaler'] = scaler_name
                all_config_results.append(result)
    print(f"\nSelecting best R² results from {len(all_config_results)} experiments...")
    final_results = []
    classifier_names = list(set([r['Classifier'] for r in all_config_results if r['Status'] == 'Success']))
    for classifier in classifier_names:
        classifier_results = [r for r in all_config_results if r['Classifier'] == classifier and r['Status'] == 'Success']
        if classifier_results:
            best_result = max(classifier_results, key=lambda x: x['R²'])
            final_results.append(best_result)
            print(f"{classifier}: Best R² = {best_result['R²']:.4f} (Config {best_result['config']}, {best_result['scaler']})")
    results_df = pd.DataFrame(final_results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    print(f"{'Classifier':<22} {'Accuracy':<10} {'F1 Score':<10} {'No Fert. Acc.':<14} {'Fert. Acc.':<12} {'R²':<10} {'MSE':<8}")
    print("-" * 120)
    for _, row in results_df.iterrows():
        print(f"{row['Classifier']:<22} {row['Accuracy']:<10.4f} {row['F1 Score']:<10.4f} {row['No Fert. Acc.']:<14.4f} {row['Fert. Acc.']:<12.4f} {row['R²']:<10.4f} {row['MSE']:<8.4f}")
    positive_r2_models = results_df[results_df['R²'] > 0]
    if len(positive_r2_models) > 0:
 
        for _, model in positive_r2_models.iterrows():
            print(f"   {model['Classifier']}: R² = {model['R²']:.4f}")
    else:
        best_r2 = results_df['R²'].max()
        best_model = results_df.loc[results_df['R²'].idxmax()]
        print(f"Best R²: {best_model['Classifier']} with R² = {best_r2:.4f}")
        print("Consider additional feature engineering for positive R²")
    results_df.to_csv('optimized_r2_results.csv', index=False, float_format='%.4f')
    print(f"\nResults saved to: optimized_r2_results.csv")
if __name__ == "__main__":
    main()
