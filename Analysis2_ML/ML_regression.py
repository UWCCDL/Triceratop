import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import warnings
warnings.filterwarnings('ignore')

data_all = pd.read_csv('combined_0.80.csv')

fa_cols = [col for col in data_all.columns if 'fa_' in col.lower()]
param_cols = ['alpha', 'proc.temp', 'decay', 'decl.temp']
available_params = [col for col in param_cols if col in data_all.columns]


X = data_all[fa_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define regression models
reg_models = {
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'LASSO (α=0.01)': Lasso(alpha=0.01, max_iter=10000),
    'LASSO (α=0.1)': Lasso(alpha=0.1, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
}

reg_results = {}

for param in available_params:
    if param in data_all.columns:
        print(f"\n{param} Prediction:")
        reg_results[param] = {}
        
        y_target = data_all[param].values
        
        for model_name, model in reg_models.items():
            try:
                cv_folds = min(5, len(y_target))
                r2_scores = cross_val_score(model, X_scaled, y_target, cv=cv_folds, scoring='r2')
                
                reg_results[param][model_name] = {
                    'r2_scores': r2_scores,
                    'mean_r2': np.mean(r2_scores),
                    'std_r2': np.std(r2_scores)
                }
                
                print(f"  {model_name}: R² = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
            except Exception as e:
                print(f"  Error with {model_name}: {e}")

# Feature importance analysis for best performing parameter
if reg_results:
    # Find parameter with best performance
    best_param = None
    best_r2 = -float('inf')
    best_model_name = None
    
    for param, models in reg_results.items():
        for model_name, results in models.items():
            if results['mean_r2'] > best_r2:
                best_r2 = results['mean_r2']
                best_param = param
                best_model_name = model_name
    
    if best_param and best_r2 > 0:
        print(f"\nBest model: {best_model_name} for {best_param}")
        print(f"R² = {best_r2:.3f}")
        
        # Fit best model for feature importance
        best_model = reg_models[best_model_name]
        best_model.fit(X_scaled, data_all[best_param].values)
        
        if hasattr(best_model, 'coef_'):
            feature_importance = pd.DataFrame({
                'Tract': fa_cols,
                'Coefficient': best_model.coef_,
                'Abs_Coefficient': np.abs(best_model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            # Filter important features
            important_features = feature_importance[feature_importance['Abs_Coefficient'] > 1e-6]
            
            if len(important_features) > 0:
                print(f"\nTop 10 important features for {best_param}:")
                print(important_features.head(10)[['Tract', 'Coefficient']])
                
                # Plot top features
                top_n = min(15, len(important_features))
                top_features = important_features.head(top_n)
                
                plt.figure(figsize=(12, 8))
                colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
                plt.barh(range(len(top_features)), top_features['Coefficient'], 
                        color=colors, alpha=0.7)
                plt.yticks(range(len(top_features)), top_features['Tract'])
                plt.xlabel('Regression Coefficient')
                plt.title(f'Important Tract Features for {best_param} Prediction')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=300)
                plt.show()
                
                # Save results
                important_features.to_csv('important_features.csv', index=False)

# Print summary
for param, models in reg_results.items():
    print(f"\n{param} Prediction:")
    for model_name, results in models.items():
        r2_val = results['mean_r2']
        if r2_val > 0.1:
            print(f"  {model_name}: R² = {r2_val:.3f} ± {results['std_r2']:.3f} *")
        else:
            print(f"  {model_name}: R² = {r2_val:.3f} ± {results['std_r2']:.3f}")

print("\n* R² > 0.1 indicates some predictive ability")

# Save summary results
if reg_results:
    summary_data = []
    for param, models in reg_results.items():
        for model_name, results in models.items():
            summary_data.append({
                'Parameter': param,
                'Model': model_name,
                'Mean_R2': results['mean_r2'],
                'Std_R2': results['std_r2']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('regression_summary.csv', index=False)
    print(f"\nResults saved to 'regression_summary.csv'")

print("\nAnalysis complete!")