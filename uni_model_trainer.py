"""
Universal Model Trainer - Works with Any Dataset
Automatically detects data types, preprocesses features, and trains multiple ML models.
Supports classification, regression, and multi-class problems.
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score, 
    precision_recall_curve, roc_curve
)

# Models for classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

warnings.filterwarnings('ignore')

class UniversalModelTrainer:
    """
    Universal model trainer that works with any dataset.
    Automatically detects problem type, preprocesses data, and trains multiple models.
    """
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2, cv_folds: int = 5):
        """
        Initialize the universal model trainer.
        
        Args:
            random_state: Random state for reproducibility
            test_size: Proportion of dataset for testing
            cv_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        
        # Dataset information
        self.problem_type = None  # 'binary_classification', 'multiclass_classification', 'regression'
        self.feature_types = {}   # {'numeric': [...], 'categorical': [...], 'datetime': [...]}
        self.target_column = None
        self.feature_columns = []
        self.class_names = []
        
        # Preprocessing components
        self.preprocessor = None
        self.label_encoder = None
        self.feature_selector = None
        
        # Models and results
        self.models = {}
        self.model_configs = {}
        self.training_results = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialize model configurations
        self._initialize_model_configs()
    
    def _initialize_model_configs(self):
        """Initialize model configurations for different problem types."""
        
        # Classification models
        self.classification_models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs']
                },
                'description': 'Linear model for classification with regularization'
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'description': 'Ensemble of decision trees with voting'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'description': 'Sequential ensemble with gradient boosting'
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'description': 'Support Vector Machine with kernel trick'
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'description': 'K-Nearest Neighbors classification'
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
                'description': 'Probabilistic classifier based on Bayes theorem'
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                },
                'description': 'Single decision tree classifier'
            },
            'mlp': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                },
                'description': 'Multi-layer Perceptron neural network'
            }
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            self.classification_models['xgboost'] = {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'description': 'Extreme Gradient Boosting'
            }
        
        if LIGHTGBM_AVAILABLE:
            self.classification_models['lightgbm'] = {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [15, 31, 63]
                },
                'description': 'Light Gradient Boosting Machine'
            }
        
        if CATBOOST_AVAILABLE:
            self.classification_models['catboost'] = {
                'model': CatBoostClassifier(random_state=self.random_state, verbose=False),
                'params': {
                    'iterations': [50, 100, 200],
                    'depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'description': 'Categorical Boosting'
            }
        
        # Regression models
        self.regression_models = {
            'linear_regression': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10, 100]
                },
                'description': 'Linear regression with L2 regularization'
            },
            'lasso': {
                'model': Lasso(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10, 100]
                },
                'description': 'Linear regression with L1 regularization'
            },
            'elastic_net': {
                'model': ElasticNet(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                'description': 'Linear regression with L1 and L2 regularization'
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                },
                'description': 'Random Forest for regression'
            },
            'svr': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'description': 'Support Vector Regression'
            },
            'knn_regressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                'description': 'K-Nearest Neighbors regression'
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=self.random_state),
                'params': {
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                },
                'description': 'Decision tree for regression'
            },
            'mlp': {
                'model': MLPRegressor(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'description': 'Multi-layer Perceptron for regression'
            }
        }
        
        # Add advanced regression models
        if XGBOOST_AVAILABLE:
            self.regression_models['xgboost'] = {
                'model': xgb.XGBRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'description': 'XGBoost for regression'
            }
        
        if LIGHTGBM_AVAILABLE:
            self.regression_models['lightgbm'] = {
                'model': lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'description': 'LightGBM for regression'
            }
    
    def analyze_dataset(self, data: Union[str, pd.DataFrame], target_column: str = None) -> Dict:
        """
        Automatically analyze dataset to determine problem type and data characteristics.
        
        Args:
            data: Path to CSV file or pandas DataFrame
            target_column: Name of target column (will auto-detect if None)
            
        Returns:
            Dictionary containing dataset analysis results
        """
        print("🔍 Analyzing dataset...")
        
        # Load data if path provided
        if isinstance(data, str):
            df = pd.read_csv(data)
            print(f"📊 Loaded dataset from {data}")
        else:
            df = data.copy()
        
        print(f"📈 Dataset shape: {df.shape}")
        
        # Auto-detect target column if not provided
        if target_column is None:
            target_column = self._detect_target_column(df)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        self.target_column = target_column
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Analyze target variable to determine problem type
        self.problem_type = self._determine_problem_type(y)
        print(f"🎯 Problem type detected: {self.problem_type}")
        
        # Analyze feature types
        self.feature_types = self._analyze_feature_types(X)
        self.feature_columns = X.columns.tolist()
        
        # Target analysis
        target_analysis = self._analyze_target(y)
        
        # Dataset quality analysis
        quality_analysis = self._analyze_data_quality(df)
        
        analysis_results = {
            'dataset_info': {
                'shape': df.shape,
                'n_features': len(self.feature_columns),
                'n_samples': len(df),
                'target_column': target_column,
                'problem_type': self.problem_type
            },
            'feature_analysis': self.feature_types,
            'target_analysis': target_analysis,
            'quality_analysis': quality_analysis
        }
        
        self._print_analysis_summary(analysis_results)
        
        return analysis_results
    
    def _detect_target_column(self, df: pd.DataFrame) -> str:
        """Auto-detect the target column."""
        possible_targets = [
            'target', 'label', 'class', 'y', 'output', 'prediction',
            'result', 'outcome', 'response', 'dependent'
        ]
        
        # Check for exact matches
        for col in possible_targets:
            if col in df.columns:
                return col
        
        # Check for partial matches
        for col in df.columns:
            col_lower = col.lower()
            for target in possible_targets:
                if target in col_lower:
                    return col
        
        # Default to last column
        return df.columns[-1]
    
    def _determine_problem_type(self, y: pd.Series) -> str:
        """Determine if this is classification or regression."""
        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(y):
            unique_values = y.nunique()
            
            # If few unique values, likely classification
            if unique_values <= 20:
                if unique_values == 2:
                    return 'binary_classification'
                else:
                    return 'multiclass_classification'
            else:
                return 'regression'
        else:
            # Categorical target
            unique_values = y.nunique()
            if unique_values == 2:
                return 'binary_classification'
            else:
                return 'multiclass_classification'
    
    def _analyze_feature_types(self, X: pd.DataFrame) -> Dict:
        """Analyze and categorize feature types."""
        feature_types = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'boolean': []
        }
        
        for col in X.columns:
            col_data = X[col]
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                feature_types['datetime'].append(col)
            
            # Check for boolean
            elif col_data.nunique() == 2 and set(col_data.dropna().unique()).issubset({0, 1, True, False}):
                feature_types['boolean'].append(col)
            
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(col_data):
                feature_types['numeric'].append(col)
            
            # Check for text (high cardinality string)
            elif pd.api.types.is_string_dtype(col_data) and col_data.nunique() > 50:
                feature_types['text'].append(col)
            
            # Everything else is categorical
            else:
                feature_types['categorical'].append(col)
        
        return feature_types
    
    def _analyze_target(self, y: pd.Series) -> Dict:
        """Analyze target variable."""
        analysis = {
            'dtype': str(y.dtype),
            'unique_values': y.nunique(),
            'missing_values': y.isnull().sum(),
            'missing_percentage': y.isnull().sum() / len(y) * 100
        }
        
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            value_counts = y.value_counts()
            analysis['class_distribution'] = value_counts.to_dict()
            analysis['class_balance'] = {
                'most_common': value_counts.max(),
                'least_common': value_counts.min(),
                'balance_ratio': value_counts.min() / value_counts.max()
            }
            self.class_names = value_counts.index.tolist()
        else:
            analysis['statistics'] = {
                'mean': y.mean(),
                'std': y.std(),
                'min': y.min(),
                'max': y.max(),
                'median': y.median()
            }
        
        return analysis
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze data quality issues."""
        analysis = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Check for high cardinality categorical features
        high_cardinality = {}
        for col in df.select_dtypes(include=['object']):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                high_cardinality[col] = df[col].nunique()
        
        analysis['high_cardinality_features'] = high_cardinality
        
        return analysis
    
    def _print_analysis_summary(self, analysis: Dict):
        """Print a summary of the dataset analysis."""
        print("\n" + "="*60)
        print("📊 DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        info = analysis['dataset_info']
        print(f"📋 Dataset: {info['n_samples']} samples, {info['n_features']} features")
        print(f"🎯 Target: {info['target_column']} ({info['problem_type']})")
        
        features = analysis['feature_analysis']
        print(f"🔢 Numeric features: {len(features['numeric'])}")
        print(f"📝 Categorical features: {len(features['categorical'])}")
        print(f"📅 DateTime features: {len(features['datetime'])}")
        print(f"📄 Text features: {len(features['text'])}")
        print(f"✅ Boolean features: {len(features['boolean'])}")
        
        quality = analysis['quality_analysis']
        print(f"💾 Memory usage: {quality['memory_usage_mb']:.1f} MB")
        print(f"🔄 Duplicate rows: {quality['duplicate_rows']} ({quality['duplicate_percentage']:.1f}%)")
        
        # Missing values summary
        missing_cols = {k: v for k, v in quality['missing_values'].items() if v > 0}
        if missing_cols:
            print(f"⚠️  Missing values: {len(missing_cols)} columns affected")
        
        target_info = analysis['target_analysis']
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            balance_ratio = target_info['class_balance']['balance_ratio']
            if balance_ratio < 0.1:
                print(f"⚠️  Class imbalance detected: {balance_ratio:.2f} ratio")
            print(f"📊 Classes: {list(target_info['class_distribution'].keys())}")
        
        print("="*60)
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, fit_preprocessor: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatically preprocess features based on their types.
        
        Args:
            X: Feature matrix
            y: Target vector  
            fit_preprocessor: Whether to fit the preprocessor (True for training data)
            
        Returns:
            Tuple of preprocessed (X, y)
        """
        print("🔧 Preprocessing data...")
        
        if fit_preprocessor:
            # Create preprocessing pipelines for different feature types
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combine transformers
            transformers = []
            
            if self.feature_types['numeric']:
                transformers.append(('num', numeric_transformer, self.feature_types['numeric']))
            
            if self.feature_types['categorical']:
                transformers.append(('cat', categorical_transformer, self.feature_types['categorical']))
            
            # Handle boolean features as numeric
            if self.feature_types['boolean']:
                bool_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('bool', bool_transformer, self.feature_types['boolean']))
            
            # Handle datetime features (extract useful components)
            if self.feature_types['datetime']:
                # For now, skip datetime features or convert to numeric
                print("⚠️  Datetime features detected but not fully supported yet")
            
            # Skip text features for now (would need NLP preprocessing)
            if self.feature_types['text']:
                print("⚠️  Text features detected but not supported in this version")
            
            if transformers:
                self.preprocessor = ColumnTransformer(
                    transformers=transformers,
                    remainder='drop'  # Drop any remaining columns
                )
                
                X_processed = self.preprocessor.fit_transform(X)
            else:
                # Fallback: basic preprocessing
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(imputer.fit_transform(X))
                
                # Store as simple pipeline
                self.preprocessor = Pipeline([
                    ('imputer', imputer),
                    ('scaler', scaler)
                ])
        else:
            # Transform using fitted preprocessor
            X_processed = self.preprocessor.transform(X)
        
        # Preprocess target variable
        y_processed = self._preprocess_target(y, fit=fit_preprocessor)
        
        print(f"✅ Preprocessing complete: {X_processed.shape} features")
        
        return X_processed, y_processed
    
    def _preprocess_target(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """Preprocess target variable."""
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            # Encode categorical targets
            if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
                if fit:
                    self.label_encoder = LabelEncoder()
                    y_processed = self.label_encoder.fit_transform(y)
                    self.class_names = self.label_encoder.classes_.tolist()
                else:
                    y_processed = self.label_encoder.transform(y)
            else:
                y_processed = y.values
                if fit:
                    self.class_names = sorted(y.unique())
        else:
            # Regression - keep as is but handle missing values
            y_processed = y.fillna(y.median()).values
        
        return y_processed
    
    def train_models(self, data: Union[str, pd.DataFrame], target_column: str = None, 
                    models_to_train: List[str] = None, use_grid_search: bool = True,
                    feature_selection: bool = False, n_features: int = None) -> Dict:
        """
        Train multiple models on the dataset.
        
        Args:
            data: Dataset path or DataFrame
            target_column: Target column name
            models_to_train: List of specific models to train (None = all)
            use_grid_search: Whether to use hyperparameter optimization
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select (None = auto)
            
        Returns:
            Dictionary containing training results
        """
        # Analyze dataset
        analysis = self.analyze_dataset(data, target_column)
        
        # Load and split data
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if self.problem_type != 'regression' else None
        )
        
        print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test")
        
        # Preprocess data
        X_train_processed, y_train_processed = self.preprocess_data(X_train, y_train, fit_preprocessor=True)
        X_test_processed, y_test_processed = self.preprocess_data(X_test, y_test, fit_preprocessor=False)
        
        # Feature selection
        if feature_selection:
            X_train_processed, X_test_processed = self._perform_feature_selection(
                X_train_processed, y_train_processed, X_test_processed, n_features
            )
        
        # Select models based on problem type
        if self.problem_type == 'regression':
            available_models = self.regression_models
        else:
            available_models = self.classification_models
        
        # Filter models if specified
        if models_to_train:
            available_models = {k: v for k, v in available_models.items() if k in models_to_train}
        
        print(f"🤖 Training {len(available_models)} models...")
        
        # Train each model
        for model_name, config in available_models.items():
            print(f"\n🔄 Training {model_name}...")
            
            try:
                result = self._train_single_model(
                    model_name, config, 
                    X_train_processed, y_train_processed,
                    use_grid_search
                )
                self.training_results[model_name] = result
                
                # Evaluate on test set
                eval_result = self._evaluate_model(
                    model_name, X_test_processed, y_test_processed
                )
                self.evaluation_results[model_name] = eval_result
                
                print(f"✅ {model_name}: {self._get_primary_metric(eval_result):.4f}")
                
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {str(e)}")
                continue
        
        # Find best model
        self._find_best_model()
        
        # Store test data for later use
        self.X_test = X_test_processed
        self.y_test = y_test_processed
        
        return {
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results,
            'best_model': self.best_model_name,
            'dataset_analysis': analysis
        }
    
    def _train_single_model(self, model_name: str, config: Dict, 
                           X_train: np.ndarray, y_train: np.ndarray,
                           use_grid_search: bool) -> Dict:
        """Train a single model with optional hyperparameter tuning."""
        start_time = datetime.now()
        
        base_model = config['model']
        param_grid = config['params']
        
        if use_grid_search and param_grid:
            # Hyperparameter optimization
            cv = StratifiedKFold(n_splits=self.cv_folds) if self.problem_type != 'regression' else KFold(n_splits=self.cv_folds)
            
            scoring = self._get_scoring_metric()
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv, scoring=scoring,
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
        else:
            # Train with default parameters
            best_model = base_model
            best_model.fit(X_train, y_train)
            best_params = {}
            best_score = None
        
        # Store trained model
        self.models[model_name] = best_model
        
        # Cross-validation scores
        cv = StratifiedKFold(n_splits=self.cv_folds) if self.problem_type != 'regression' else KFold(n_splits=self.cv_folds)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=self._get_scoring_metric())
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'best_params': best_params,
            'best_cv_score': best_score,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'model_description': config['description']
        }
    
    def _evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate a trained model on test data."""
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        if self.problem_type == 'regression':
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'predictions': y_pred.tolist()
            }
        
        else:
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                
                if self.problem_type == 'binary_classification':
                    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    # Multi-class AUC (one-vs-rest)
                    try:
                        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    except:
                        auc_score = None
            else:
                y_pred_proba = None
                auc_score = None
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'auc_score': auc_score,
                'predictions': y_pred.tolist(),
                'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
    
    def _perform_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  X_test: np.ndarray, n_features: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature selection."""
        if n_features is None:
            n_features = min(20, X_train.shape[1] // 2)  # Select top 20 or half of features
        
        print(f"🎯 Selecting top {n_features} features...")
        
        # Choose appropriate scoring function
        score_func = f_classif if self.problem_type != 'regression' else f_regression
        
        self.feature_selector = SelectKBest(score_func=score_func, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        return X_train_selected, X_test_selected
    
    def _get_scoring_metric(self) -> str:
        """Get appropriate scoring metric for the problem type."""
        if self.problem_type == 'binary_classification':
            return 'f1'
        elif self.problem_type == 'multiclass_classification':
            return 'f1_macro'
        else:
            return 'neg_mean_squared_error'
    
    def _get_primary_metric(self, eval_result: Dict) -> float:
        """Get the primary metric value for comparison."""
        if self.problem_type == 'regression':
            return eval_result['r2_score']
        else:
            return eval_result['accuracy']
    
    def _find_best_model(self):
        """Find the best performing model."""
        if not self.evaluation_results:
            return
        
        best_score = -float('inf')
        
        for model_name, eval_result in self.evaluation_results.items():
            score = self._get_primary_metric(eval_result)
            
            if score > best_score:
                best_score = score
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"🏆 Best model: {self.best_model_name} (score: {best_score:.4f})")
    
    def plot_model_comparison(self, save_path: str = None):
        """Plot comparison of model performance."""
        if not self.evaluation_results:
            print("No evaluation results to plot")
            return
        
        # Prepare data for plotting
        model_names = list(self.evaluation_results.keys())
        
        if self.problem_type == 'regression':
            metrics = ['r2_score', 'rmse', 'mae']
            metric_values = {
                'r2_score': [self.evaluation_results[m]['r2_score'] for m in model_names],
                'rmse': [self.evaluation_results[m]['rmse'] for m in model_names],
                'mae': [self.evaluation_results[m]['mae'] for m in model_names]
            }
        else:
            metrics = ['accuracy']
            metric_values = {
                'accuracy': [self.evaluation_results[m]['accuracy'] for m in model_names]
            }
            
            # Add AUC if available
            if all('auc_score' in self.evaluation_results[m] and 
                   self.evaluation_results[m]['auc_score'] is not None for m in model_names):
                metrics.append('auc_score')
                metric_values['auc_score'] = [self.evaluation_results[m]['auc_score'] for m in model_names]
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(model_names, metric_values[metric])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str = None, top_n: int = 15):
        """Plot feature importance for tree-based models."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            print(f"Feature importance not available for {model_name}")
            return
        
        # Get feature names (approximate since we may have transformed features)
        n_features = len(importance)
        
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except:
                feature_names = [f'feature_{i}' for i in range(n_features)]
        else:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def generate_report(self, save_path: str = None) -> Dict:
        """Generate comprehensive training report."""
        if not self.evaluation_results:
            print("No evaluation results available")
            return {}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'problem_type': self.problem_type,
                'n_features': len(self.feature_columns),
                'target_column': self.target_column,
                'feature_types': self.feature_types
            },
            'models_trained': list(self.models.keys()),
            'best_model': self.best_model_name,
            'training_results': self.training_results,
            'evaluation_results': {}
        }
        
        # Format evaluation results for report
        for model_name, eval_result in self.evaluation_results.items():
            if self.problem_type == 'regression':
                report['evaluation_results'][model_name] = {
                    'r2_score': eval_result['r2_score'],
                    'rmse': eval_result['rmse'],
                    'mae': eval_result['mae']
                }
            else:
                report['evaluation_results'][model_name] = {
                    'accuracy': eval_result['accuracy'],
                    'auc_score': eval_result.get('auc_score'),
                    'precision': eval_result['classification_report']['macro avg']['precision'],
                    'recall': eval_result['classification_report']['macro avg']['recall'],
                    'f1_score': eval_result['classification_report']['macro avg']['f1-score']
                }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Report saved to {save_path}")
        
        return report
    
    def save_models(self, save_dir: str = 'models/'):
        """Save all trained models and preprocessing components."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = save_path / f'{model_name}_model.joblib'
            joblib.dump(model, model_path)
        
        # Save preprocessor
        if self.preprocessor:
            joblib.dump(self.preprocessor, save_path / 'preprocessor.joblib')
        
        # Save label encoder
        if self.label_encoder:
            joblib.dump(self.label_encoder, save_path / 'label_encoder.joblib')
        
        # Save feature selector
        if self.feature_selector:
            joblib.dump(self.feature_selector, save_path / 'feature_selector.joblib')
        
        # Save metadata
        metadata = {
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'feature_types': self.feature_types,
            'class_names': self.class_names,
            'best_model': self.best_model_name,
            'models_available': list(self.models.keys()),
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Models saved to {save_dir}")
        print(f"   - {len(self.models)} model files")
        print(f"   - Preprocessor and metadata")
    
    def load_models(self, save_dir: str = 'models/'):
        """Load previously trained models."""
        save_path = Path(save_dir)
        
        if not save_path.exists():
            raise FileNotFoundError(f"Models directory {save_dir} not found")
        
        # Load metadata
        with open(save_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.problem_type = metadata['problem_type']
        self.target_column = metadata['target_column']
        self.feature_columns = metadata['feature_columns']
        self.feature_types = metadata['feature_types']
        self.class_names = metadata['class_names']
        self.best_model_name = metadata['best_model']
        
        # Load models
        for model_name in metadata['models_available']:
            model_path = save_path / f'{model_name}_model.joblib'
            if model_path.exists():
                model = joblib.load(model_path)
                # Fix for newer XGBoost versions that removed use_label_encoder
                if hasattr(model, 'use_label_encoder'):
                    del model.use_label_encoder
                self.models[model_name] = model
        
        # Load preprocessor
        preprocessor_path = save_path / 'preprocessor.joblib'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
        
        # Load label encoder
        label_encoder_path = save_path / 'label_encoder.joblib'
        if label_encoder_path.exists():
            self.label_encoder = joblib.load(label_encoder_path)
        
        # Load feature selector
        feature_selector_path = save_path / 'feature_selector.joblib'
        if feature_selector_path.exists():
            self.feature_selector = joblib.load(feature_selector_path)
        
        self.best_model = self.models.get(self.best_model_name)
        
        print(f"📂 Loaded models from {save_dir}")
        print(f"   - {len(self.models)} models loaded")
        print(f"   - Best model: {self.best_model_name}")
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray], model_name: str = None) -> Dict:
        """
        Make predictions on new data.
        
        Args:
            data: New data to predict on
            model_name: Specific model to use (None = best model)
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Preprocess data
        if isinstance(data, pd.DataFrame):
            # Ensure columns match training data
            data_processed, _ = self.preprocess_data(data, pd.Series([0]*len(data)), fit_preprocessor=False)
        else:
            data_processed = data
        
        # Apply feature selection if used
        if self.feature_selector:
            data_processed = self.feature_selector.transform(data_processed)
        
        # Make predictions
        predictions = model.predict(data_processed)
        
        result = {
            'predictions': predictions.tolist(),
            'model_used': model_name
        }
        
        # Add probabilities for classification
        if self.problem_type != 'regression' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(data_processed)
            result['probabilities'] = probabilities.tolist()
            
            if self.label_encoder:
                # Convert numeric predictions back to original labels
                original_predictions = self.label_encoder.inverse_transform(predictions.astype(int))
                result['original_predictions'] = original_predictions.tolist()
        
        return result
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models."""
        if not self.evaluation_results:
            print("No models trained yet")
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name in self.models.keys():
            row = {'model': model_name}
            
            # Training info
            if model_name in self.training_results:
                train_result = self.training_results[model_name]
                row.update({
                    'cv_score': train_result['cv_mean'],
                    'cv_std': train_result['cv_std'],
                    'training_time': train_result['training_time']
                })
            
            # Evaluation info
            if model_name in self.evaluation_results:
                eval_result = self.evaluation_results[model_name]
                
                if self.problem_type == 'regression':
                    row.update({
                        'r2_score': eval_result['r2_score'],
                        'rmse': eval_result['rmse'],
                        'mae': eval_result['mae']
                    })
                else:
                    row.update({
                        'accuracy': eval_result['accuracy'],
                        'auc_score': eval_result.get('auc_score', np.nan)
                    })
            
            # Mark best model
            row['is_best'] = model_name == self.best_model_name
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data).sort_values('cv_score', ascending=False)

# Convenience functions for common use cases
def quick_train(data_path: str, target_column: str = None, models: List[str] = None) -> UniversalModelTrainer:
    """
    Quick training function for immediate results.
    
    Args:
        data_path: Path to CSV dataset
        target_column: Target column name (auto-detect if None)
        models: List of models to train (None = all)
        
    Returns:
        Trained UniversalModelTrainer instance
    """
    trainer = UniversalModelTrainer()
    
    # Fast training without grid search
    results = trainer.train_models(
        data=data_path,
        target_column=target_column,
        models_to_train=models,
        use_grid_search=False
    )
    
    # Save models
    trainer.save_models()
    
    # Print summary
    summary = trainer.get_model_summary()
    print("\n🏆 QUICK TRAINING SUMMARY")
    print("="*50)
    print(summary.to_string(index=False))
    
    return trainer

def comprehensive_train(data_path: str, target_column: str = None, 
                       feature_selection: bool = True) -> UniversalModelTrainer:
    """
    Comprehensive training with all optimizations.
    
    Args:
        data_path: Path to CSV dataset
        target_column: Target column name
        feature_selection: Whether to perform feature selection
        
    Returns:
        Trained UniversalModelTrainer instance
    """
    trainer = UniversalModelTrainer()
    
    # Full training with hyperparameter optimization
    results = trainer.train_models(
        data=data_path,
        target_column=target_column,
        use_grid_search=True,
        feature_selection=feature_selection
    )
    
    # Generate visualizations
    trainer.plot_model_comparison('model_comparison.png')
    trainer.plot_feature_importance()
    
    # Save everything
    trainer.save_models()
    trainer.generate_report('training_report.json')
    
    return trainer

def benchmark_models_on_dataset(data_path: str, target_column: str = None) -> pd.DataFrame:
    """
    Quick benchmark of all models on a dataset.
    
    Args:
        data_path: Path to CSV dataset
        target_column: Target column name
        
    Returns:
        DataFrame with model comparison results
    """
    trainer = UniversalModelTrainer()
    
    # Quick training for benchmarking
    trainer.train_models(
        data=data_path,
        target_column=target_column,
        use_grid_search=False
    )
    
    return trainer.get_model_summary()

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal Model Trainer')
    parser.add_argument('data_path', help='Path to CSV dataset')
    parser.add_argument('--target', help='Target column name (auto-detect if not provided)')
    parser.add_argument('--models', nargs='+', help='Specific models to train')
    parser.add_argument('--quick', action='store_true', help='Quick training without grid search')
    parser.add_argument('--output-dir', default='models/', help='Output directory for models')
    parser.add_argument('--feature-selection', action='store_true', help='Perform feature selection')
    
    args = parser.parse_args()
    
    print("🚀 Universal Model Trainer")
    print("="*50)
    
    # Initialize trainer
    trainer = UniversalModelTrainer()
    
    try:
        # Train models
        results = trainer.train_models(
            data=args.data_path,
            target_column=args.target,
            models_to_train=args.models,
            use_grid_search=not args.quick,
            feature_selection=args.feature_selection
        )
        
        # Display results
        summary = trainer.get_model_summary()
        print("\n📊 TRAINING RESULTS")
        print("="*50)
        print(summary.to_string(index=False))
        
        # Save models
        trainer.save_models(args.output_dir)
        
        # Generate visualizations
        trainer.plot_model_comparison()
        
        if trainer.best_model_name:
            trainer.plot_feature_importance()
        
        # Generate report
        report_path = Path(args.output_dir) / 'training_report.json'
        trainer.generate_report(str(report_path))
        
        print(f"\n🎉 Training complete! Best model: {trainer.best_model_name}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise