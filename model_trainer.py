"""
Model Trainer for Phishing Website Detection
Trains and evaluates machine learning models with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PhishingModelTrainer:
    """
    Comprehensive model trainer for phishing detection.
    Supports multiple algorithms with hyperparameter optimization.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        self.training_history = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def load_data(self, data_path, target_column='label'):
        """
        Load and preprocess training data.
        
        Args:
            data_path (str): Path to CSV file with features and labels
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column, 'url'], errors='ignore')
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(-1)  # Fill with -1 to indicate missing data
        
        # Encode labels if they are strings
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name, X_train, y_train, use_grid_search=True):
        """
        Train a specific model with optional hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            use_grid_search (bool): Whether to use grid search for hyperparameters
            
        Returns:
            dict: Training results and metrics
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model '{model_name}' not supported")
        
        print(f"\nTraining {model_name}...")
        start_time = datetime.now()
        
        # Get model configuration
        config = self.model_configs[model_name]
        base_model = config['model']
        param_grid = config['params']
        
        # Scale features for neural network
        if model_name == 'neural_network':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
            self.scalers[model_name] = None
        
        # Hyperparameter tuning
        if use_grid_search:
            print("  Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='f1',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
        else:
            print("  Training with default parameters...")
            best_model = base_model
            best_model.fit(X_train_scaled, y_train)
            best_params = {}
            best_score = -1
        
        # Store the trained model
        self.models[model_name] = best_model
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X_train_scaled, y_train, cv=5, scoring='f1'
        )
        
        # Store training history
        self.training_history[model_name] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'trained_at': datetime.now().isoformat()
        }
        
        print(f"  Best parameters: {best_params}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Training time: {training_time:.2f} seconds")
        
        return self.training_history[model_name]
    
    def train_all_models(self, X_train, y_train, use_grid_search=True):
        """
        Train all configured models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            use_grid_search (bool): Whether to use grid search
            
        Returns:
            dict: Training results for all models
        """
        print("Training all models...")
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                result = self.train_model(model_name, X_train, y_train, use_grid_search)
                results[model_name] = result
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model on test data.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Scale features if needed
        if scaler:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        evaluation_results = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_score': auc_score,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        print(f"\n{model_name} Evaluation Results:")
        print(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"  Precision: {evaluation_results['precision']:.4f}")
        print(f"  Recall: {evaluation_results['recall']:.4f}")
        print(f"  F1-Score: {evaluation_results['f1_score']:.4f}")
        print(f"  AUC Score: {evaluation_results['auc_score']:.4f}")
        
        return evaluation_results
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Evaluation results for all models
        """
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = result
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return results
    
    def get_feature_importance(self, model_name, top_n=20):
        """
        Get feature importance for a trained model.
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.models[model_name]
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            print(f"Feature importance not available for {model_name}")
            return None
        
        # Create dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def plot_model_comparison(self, evaluation_results):
        """
        Plot comparison of model performance.
        
        Args:
            evaluation_results (dict): Results from evaluate_all_models()
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        model_names = list(evaluation_results.keys())
        
        # Create comparison dataframe
        comparison_data = []
        for model_name in model_names:
            row = {'model': model_name}
            for metric in metrics:
                row[metric] = evaluation_results[model_name][metric]
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(df_comparison['model'], df_comparison[metric])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model_name, top_n=15):
        """
        Plot feature importance for a model.
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to plot
        """
        importance_df = self.get_feature_importance(model_name, top_n)
        
        if importance_df is None:
            return
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, save_dir='models/'):
        """
        Save all trained models and related objects.
        
        Args:
            save_dir (str): Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        scalers_path = os.path.join(save_dir, 'scalers.joblib')
        joblib.dump(self.scalers, scalers_path)
        
        # Save other objects
        metadata = {
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
        }
        
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model training complete. Files saved to {save_dir}")
    
    def load_models(self, save_dir='models/'):
        """
        Load previously trained models.
        
        Args:
            save_dir (str): Directory containing saved models
        """
        import os
        
        # Load metadata
        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_history = metadata['training_history']
        
        # Load label encoder
        if metadata['label_encoder_classes']:
            self.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
        
        # Load scalers
        scalers_path = os.path.join(save_dir, 'scalers.joblib')
        self.scalers = joblib.load(scalers_path)
        
        # Load models
        for model_name in self.model_configs.keys():
            model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} from {model_path}")

def create_sample_dataset():
    """
    Create a sample dataset for testing.
    This would normally be replaced with actual phishing data.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    features = {
        'url_length': np.random.normal(50, 30, n_samples),
        'domain_length': np.random.normal(15, 8, n_samples),
        'has_https': np.random.binomial(1, 0.7, n_samples),
        'has_ip_address': np.random.binomial(1, 0.1, n_samples),
        'subdomain_count': np.random.poisson(1, n_samples),
        'dots_count': np.random.poisson(3, n_samples),
        'domain_age': np.random.exponential(365, n_samples),
        'has_ssl': np.random.binomial(1, 0.8, n_samples)
    }
    
    df = pd.DataFrame(features)
    
    # Create labels based on rules (phishing = 1, legitimate = 0)
    conditions = (
        (df['url_length'] > 100) |
        (df['has_ip_address'] == 1) |
        (df['has_https'] == 0) |
        (df['domain_age'] < 30) |
        (df['subdomain_count'] > 3)
    )
    
    df['label'] = conditions.astype(int)
    
    # Add some URLs for reference
    df['url'] = [f'http://example{i}.com' for i in range(n_samples)]
    
    return df

if __name__ == "__main__":
    # Example usage
    print("Creating sample dataset...")
    sample_data = create_sample_dataset()
    sample_data.to_csv('sample_phishing_data.csv', index=False)
    
    # Initialize trainer
    trainer = PhishingModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data('sample_phishing_data.csv')
    
    # Train models
    print("\nTraining models...")
    training_results = trainer.train_all_models(X_train, y_train, use_grid_search=False)
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluation_results = trainer.evaluate_all_models(X_test, y_test)
    
    # Plot comparisons
    trainer.plot_model_comparison(evaluation_results)
    
    # Show feature importance
    for model_name in trainer.models.keys():
        trainer.plot_feature_importance(model_name)
    
    # Save models
    trainer.save_models()