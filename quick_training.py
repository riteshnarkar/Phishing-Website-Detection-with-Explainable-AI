import sys
sys.path.append('src')
from model_trainer import PhishingModelTrainer

# Initialize trainer
trainer = PhishingModelTrainer()

# Load data
print('📊 Loading training data...')
X_train, X_test, y_train, y_test = trainer.load_data('data/processed/features_dataset.csv')

# Train all models (no hyperparameter tuning for speed)
print('🤖 Training models...')
training_results = trainer.train_all_models(X_train, y_train, use_grid_search=False)

# Evaluate models
print('📈 Evaluating models...')
evaluation_results = trainer.evaluate_all_models(X_test, y_test)

# Save models
print('💾 Saving trained models...')
trainer.save_models('models/')

print('✅ Model training complete!')