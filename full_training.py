import sys
sys.path.append('src')
from model_trainer import PhishingModelTrainer

trainer = PhishingModelTrainer()
X_train, X_test, y_train, y_test = trainer.load_data('data/dataset_features_synthetic.csv')

print('⚡ Training with grid search disabled for speed...')
# Override XGBoost defaults with better hyperparameters before training
import xgboost as xgb
trainer.model_configs['xgboost']['model'] = xgb.XGBClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss'
)

training_results = trainer.train_all_models(X_train, y_train, use_grid_search=False)
evaluation_results = trainer.evaluate_all_models(X_test, y_test)
trainer.save_models('models/')
print('✅ Training complete!')

# Show feature importance for XGBoost
importance_df = trainer.get_feature_importance('xgboost', top_n=46)
if importance_df is not None:
    print('\n📊 XGBoost Feature Importance (top 20):')
    for i, row in importance_df.head(20).iterrows():
        print(f'  {row["feature"]:<30s}: {row["importance"]:.4f}')
    nonzero = (importance_df['importance'] > 0).sum()
    print(f'\nFeatures with non-zero importance: {nonzero}/{len(importance_df)}')