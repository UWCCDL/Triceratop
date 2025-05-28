import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

model_all = pd.read_csv('LL_model2_0.80.csv')
dti_data = pd.read_csv('bundle_profiles_wide.csv')

model_all['subject_numeric'] = model_all['HCPID'].str.extract(r'(\d+)').astype(int)
dti_data['subject_numeric'] = dti_data['subject']

fa_cols = [col for col in dti_data.columns if '_FA' in col]
#tract_names = list(set([col.split('_')[0] for col in fa_cols]))
tract_names = ["AntFrontal","ARC_L","ARC_R","ATR_L","ATR_R","CGC_L","CGC_R",
                "CST_L","CST_R","IFO_L","IFO_R","ILF_L","ILF_R","Motor",
                "Occipital","Orbital","PostParietal","SLF_L","SLF_R",
                "SupFrontal","SupParietal","Temporal","UNC_L","UNC_R"]

mean_fa_data = pd.DataFrame({'subject_numeric': dti_data['subject_numeric'].unique()})

# mean FA for each tract and subject
for tract in tract_names:
    tract_cols = [col for col in fa_cols if col.startswith(f"{tract}_")]
    mean_fa_data[f'fa_{tract}'] = dti_data.groupby('subject_numeric')[tract_cols].mean().mean(axis=1).values

# merge data
data_all = pd.merge(mean_fa_data, model_all, on='subject_numeric', how='inner')
data_all.to_csv('combined_0.80.csv', index=False)

print(f"Dataset shape: {data_all.shape}")

# # QC filter (80% uniform response threshold)
# uniform_threshold = 0.80
# # # data_all = data_all[data_all['uniform_response_rate'] < uniform_threshold]
# print(f"After QC filtering: {data_all.shape}")


fa_cols = [col for col in data_all.columns if 'fa_' in col.lower()]
print(f"{len(fa_cols)} tract features")

X = data_all[fa_cols].values
y_class = (data_all['best.model'] == 'Procedural').astype(int)

print(f"Feature matrix shape: {X.shape}")

# class distribution
class_counts = data_all['best.model'].value_counts()
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} ({100*count/len(data_all):.1f}%)")

class_balance = np.mean(y_class)
print(class_balance)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use stratified CV to maintain class balance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("5-fold stratified cross-validation")

models = {
    'Logistic Regression': LogisticRegression(penalty=None, random_state=42, max_iter=1000),
    'L1 Regularized (C=1.0)': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42),
    'L1 Regularized (C=0.1)': LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42),
    'L2 Regularized (C=1.0)': LogisticRegression(penalty='l2', C=1.0, random_state=42),
    'L2 Regularized (C=0.1)': LogisticRegression(penalty='l2', C=0.1, random_state=42),
}

# Cross-validation results
cv_results = {}
for name, model in models.items():
    try:
        accuracy_scores = cross_val_score(model, X_scaled, y_class, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(model, X_scaled, y_class, cv=cv, scoring='roc_auc')
        
        cv_results[name] = {
            'accuracy': accuracy_scores,
            'auc': auc_scores,
            'mean_acc': np.mean(accuracy_scores),
            'std_acc': np.std(accuracy_scores),
            'mean_auc': np.mean(auc_scores),
            'std_auc': np.std(auc_scores)
        }
        print(f"{name}: Accuracy = {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}, "
              f"AUC = {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    except Exception as e:
        print(f"Error with {name}: {e}")


if cv_results:
    # Select best model based on AUC
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_auc'])
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name}")
    print(f"AUC: {cv_results[best_model_name]['mean_auc']:.3f} ± {cv_results[best_model_name]['std_auc']:.3f}")
    
    # Fit best model and generate detailed report
    best_model.fit(X_scaled, y_class)
    y_pred = best_model.predict(X_scaled)
    y_pred_proba = best_model.predict_proba(X_scaled)[:, 1]
    
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_class, y_pred, target_names=['Declarative', 'Procedural']))
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_class, y_pred_proba, name=f'{best_model_name}')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.title(f'ROC Curve: {best_model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('classification_roc_curve.png', dpi=300)
    plt.show()

#---------Permutation Test---------------------

if cv_results and cv_results[best_model_name]['mean_auc'] > 0.55:
    print("Running permutation test...")
    score, perm_scores, pvalue = permutation_test_score(
        best_model, X_scaled, y_class, scoring='roc_auc', cv=cv, 
        n_permutations=1000, random_state=42
    )
    
    print(f"Permutation test p-value: {pvalue:.4f}")
    print(f"True AUC: {score:.3f}")
    print(f"Chance level AUC: {np.mean(perm_scores):.3f} ± {np.std(perm_scores):.3f}")
    
    # Plot permutation test results
    plt.figure(figsize=(10, 6))
    plt.hist(perm_scores, bins=50, alpha=0.7, density=True, label=f'Permutation scores (n=1000)')
    plt.axvline(score, color='red', linestyle='--', linewidth=2, label=f'True score: {score:.3f}')
    plt.axvline(np.mean(perm_scores), color='black', linestyle='-', alpha=0.5, 
                label=f'Chance level: {np.mean(perm_scores):.3f}')
    plt.xlabel('AUC Score')
    plt.ylabel('Density')
    plt.title(f'Permutation Test Results\n{best_model_name} (p = {pvalue:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('permutation_test.png', dpi=300)
    plt.show()
else:
    print("Skipping permutation test (AUC too low)")


# Use lasso for feature selection
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso_model.fit(X_scaled, y_class)

# Get feature importance
feature_importance = pd.DataFrame({
    'Tract': fa_cols,
    'Coefficient': lasso_model.coef_[0],
    'Abs_Coefficient': np.abs(lasso_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

# Filter selected features
selected_features = feature_importance[feature_importance['Abs_Coefficient'] > 1e-6]
print(f"LASSO selected {len(selected_features)} features out of {len(fa_cols)} ")

if len(selected_features) > 0:
    # Plot top features
    top_n = min(15, len(selected_features))
    top_features = selected_features.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['Tract'])
    plt.xlabel('LASSO Coefficient')
    plt.title('Selected Tract Features for Classification')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig('classification_feature_importance.png', dpi=300)
    plt.show()
    
    # Save results
    selected_features.to_csv('classification_selected_features.csv', index=False)
    print(f"Top {min(10, len(selected_features))} selected features:")
    print(selected_features.head(10)[['Tract', 'Coefficient']])
else:
    print("No features selected by LASSO regularization")



print("Classification Summary:")
if cv_results:
    for name, results in cv_results.items():
        print(f"  {name}: {results['mean_auc']:.3f} AUC")
    
    best_auc = cv_results[best_model_name]['mean_auc']
    if best_auc < 0.6:
        print(f"\nCONCLUSION: Poor classification performance (AUC = {best_auc:.3f})")
    else:
        print(f"\nCONCLUSION: Moderate classification performance (AUC = {best_auc:.3f})")

# Save summary
if cv_results:
    results_summary = pd.DataFrame([
        (name, res['mean_auc'], res['std_auc'], res['mean_acc'], res['std_acc']) 
        for name, res in cv_results.items()
    ], columns=['Model', 'Mean_AUC', 'Std_AUC', 'Mean_Accuracy', 'Std_Accuracy'])
    results_summary.to_csv('classification_results_summary.csv', index=False)