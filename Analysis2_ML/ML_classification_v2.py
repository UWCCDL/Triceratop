import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, RocCurveDisplay, roc_curve, auc


data_all = pd.read_csv('combined_0.80.csv')
print(data_all['best.model'].value_counts())


# remove subjectss worse than random model

n_trials = 64 
random_threshold = n_trials * np.log(0.5)  # unsure if this makes sense? revisit later
print(f"Random model threshold: {random_threshold:.2f}")

data_filtered = data_all[
    (data_all['proc.LL'] > random_threshold) & 
    (data_all['decl.LL'] > random_threshold)
].copy()

data_filtered = data_filtered.reset_index(drop=True) #reset index after removing subjects

print(f"Participants removed: {len(data_all) - len(data_filtered)}")
print(f"Remaining participants: {len(data_filtered)}")

print(f"\nClass distribution after filtering:")
print(data_filtered['best.model'].value_counts())

fa_cols = [col for col in data_filtered.columns if col.startswith('fa_')]
print(f"{len(fa_cols)} FA features")

X = data_filtered[fa_cols].values
y_class = (data_filtered['best.model'] == 'Procedural').astype(int)

# calculate weights using diff.LL
# subjects with greater absolute value of diff.LL -> weighted more
model_weights = np.abs(data_filtered['diff.LL'].values)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Final class distribution
class_counts = np.bincount(y_class)
print(f"\nFinal class distribution:")
print(f"  Declarative (0): {class_counts[0]} ({100*class_counts[0]/len(y_class):.1f}%)")
print(f"  Procedural (1): {class_counts[1]} ({100*class_counts[1]/len(y_class):.1f}%)")


#=========================== Weighted LASSO check =======================================
#weights based on diff.LL
w = data_all["diff.LL"].values
W = np.sqrt(abs(model_weights))

# LASSO logistic regression
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
lasso.fit(X_scaled, y_class, sample_weight=W)

y_prob = lasso.predict_proba(X_scaled)[:,1]

RocCurveDisplay.from_predictions(
    y_class,
    y_prob,
    name=f"Validation (AUC = {roc_auc_score(y_class, y_prob):.2f})",
    color="blue"
)
plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#===========================================================================================

# class_weight = 'balanced'
models = {
    'Logistic (Balanced)': LogisticRegression(
        penalty=None, 
        class_weight='balanced', 
        random_state=42, 
        max_iter=1000
    ),
    'L1 LASSO (C=1.0, Balanced)': LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=1.0, 
        class_weight='balanced', 
        random_state=42
    ),
    'L1 LASSO (C=0.1, Balanced)': LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C=0.1, 
        class_weight='balanced', 
        random_state=42
    ),
    'L2 Ridge (Balanced)': LogisticRegression(
        penalty='l2', 
        C=1.0, 
        class_weight='balanced', 
        random_state=42
    ),
    'SVM (Balanced)': SVC(
        kernel='linear', 
        class_weight='balanced', 
        probability=True,
        random_state=42
    )
}

# cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    print(f"\nTesting {name}...")
    
    accuracy_scores = []
    auc_scores = []
    
    for train_idx, val_idx in cv.split(X_scaled, y_class):
        # Use the indices directly on the numpy arrays
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_class[train_idx], y_class[val_idx]
        weights_train = model_weights[train_idx]
        
        # Fit with sample weights
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Predict
        y_pred = model.predict(X_val)
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_pred_prob))
    
    cv_results[name] = {
        'accuracy': np.array(accuracy_scores),
        'auc': np.array(auc_scores),
        'mean_acc': np.mean(accuracy_scores),
        'std_acc': np.std(accuracy_scores),
        'mean_auc': np.mean(auc_scores),
        'std_auc': np.std(auc_scores)
    }
    
    print(f"  Accuracy: {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}")
    print(f"  AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")


best_model_name = max(cv_results.keys(), 
                     key=lambda k: cv_results[k]['mean_auc'])
best_model = models[best_model_name]

print(f"Best model: {best_model_name}")
print(f"Best AUC: {cv_results[best_model_name]['mean_auc']:.3f} ± {cv_results[best_model_name]['std_auc']:.3f}")

# Fit final model
best_model.fit(X_scaled, y_class, sample_weight=model_weights)

# Predictions
y_pred_final = best_model.predict(X_scaled)
y_pred_prob_final = best_model.predict_proba(X_scaled)[:, 1]

print(classification_report(y_class, y_pred_final, 
                          target_names=['Declarative', 'Procedural']))


#========================================================================
# Model weights distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(model_weights, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Model Evidence Weight |diff.LL|')
plt.ylabel('Frequency')
plt.title('Distribution of Model Weights')

plt.subplot(1, 3, 2)
plt.hist(data_filtered['diff.LL'], bins=30, alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Zero threshold')
plt.xlabel('Log-Likelihood Difference')
plt.ylabel('Frequency')
plt.title('Model Preference Distribution')
plt.legend()

fpr, tpr, _ = roc_curve(y_class, y_pred_prob_final)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 3, 3) # ROC Curve
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve: {best_model_name}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# feature importance
# if 'L1' in best_model_name and hasattr(best_model, 'coef_'):
#     feature_importance = pd.DataFrame({
#         'Tract': fa_cols,
#         'Coefficient': best_model.coef_[0],
#         'Abs_Coefficient': np.abs(best_model.coef_[0])
#     }).sort_values('Abs_Coefficient', ascending=False)
    
#     # Filter selected features
#     selected_features = feature_importance[feature_importance['Abs_Coefficient'] > 1e-6]
#     print(f"LASSO selected {len(selected_features)} out of {len(fa_cols)} features")
    
#     if len(selected_features) > 0:
#         # Plot feature importance
#         top_n = min(15, len(selected_features))
#         top_features = selected_features.head(top_n)
        
#         plt.figure(figsize=(12, 8))
#         colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
#         plt.barh(range(len(top_features)), top_features['Coefficient'], 
#                 color=colors, alpha=0.7)
#         plt.yticks(range(len(top_features)), 
#                   [col.replace('fa_', '') for col in top_features['Tract']])
#         plt.xlabel('LASSO Coefficient')
#         plt.title(f'Selected Features: {best_model_name}')
#         plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
#         plt.tight_layout()
#         plt.savefig('feature_importance_improved.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         print(f"Top 10 selected features:")
#         print(selected_features.head(10)[['Tract', 'Coefficient']])
        
#         # Save results
#         selected_features.to_csv('selected_features_improved.csv', index=False)


print("Model Performance Summary:")
for name, results in cv_results.items():
    auc = results['mean_auc']
    auc_std = results['std_auc']
    if auc > 0.6:
        print(f"  {name}: {auc:.3f} ± {auc_std:.3f} AUC **")
    elif auc > 0.55:
        print(f"  {name}: {auc:.3f} ± {auc_std:.3f} AUC *")
    else:
        print(f"  {name}: {auc:.3f} ± {auc_std:.3f} AUC")


print(f"\nBest model: {best_model_name}")
print(f"Best AUC: {cv_results[best_model_name]['mean_auc']:.3f}")

results_df = pd.DataFrame([
    (name, res['mean_auc'], res['std_auc'], res['mean_acc'], res['std_acc'])
    for name, res in cv_results.items()
], columns=['Model', 'Mean_AUC', 'Std_AUC', 'Mean_Accuracy', 'Std_Accuracy'])

results_df.to_csv('improved_classification_results.csv', index=False)
print(f"\nResults saved to 'improved_classification_results.csv'")