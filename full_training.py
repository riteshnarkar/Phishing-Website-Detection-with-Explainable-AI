import sys
sys.path.append('src')
from model_trainer import PhishingModelTrainer

trainer = PhishingModelTrainer()
X_train, X_test, y_train, y_test = trainer.load_data('data/dataset_features_synthetic.csv')

print('⚡ Fast Training (No grid search)...')
# Skipping grid search to prevent 30-minute hangs
training_results = trainer.train_all_models(X_train, y_train, use_grid_search=False)
evaluation_results = trainer.evaluate_all_models(X_test, y_test)
trainer.save_models('models/')
print('✅ Fast training complete!')