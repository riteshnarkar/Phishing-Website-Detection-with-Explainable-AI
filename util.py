"""
Utility Scripts for Phishing Detection Project
Includes data download, preparation, and validation utilities.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import json
import zipfile
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download_phishing_datasets():
    """
    Download publicly available phishing datasets.
    """
    datasets = {
        'uci_phishing': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff',
            'filename': 'uci_phishing.arff',
            'description': 'UCI Phishing Website Dataset'
        },
        'phishtank_sample': {
            # Note: PhishTank requires API key for full access
            'url': 'http://data.phishtank.com/data/online-valid.csv',
            'filename': 'phishtank_verified.csv',
            'description': 'PhishTank Verified Phishing URLs'
        }
    }
    
    os.makedirs('data/raw', exist_ok=True)
    
    print("Downloading phishing datasets...")
    
    for dataset_name, info in datasets.items():
        print(f"\nDownloading {info['description']}...")
        try:
            response = requests.get(info['url'], timeout=30)
            response.raise_for_status()
            
            filepath = os.path.join('data/raw', info['filename'])
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Downloaded to {filepath}")
            
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {str(e)}")

def create_sample_dataset(n_samples=5000):
    """
    Create a sample dataset for testing when real data is not available.
    """
    print(f"Creating sample dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate realistic phishing and legitimate URLs
    legitimate_domains = [
        'google.com', 'amazon.com', 'facebook.com', 'microsoft.com', 'apple.com',
        'youtube.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'github.com',
        'stackoverflow.com', 'wikipedia.org', 'reddit.com', 'netflix.com', 'paypal.com'
    ]
    
    phishing_patterns = [
        'secure-{domain}', '{domain}-update', '{domain}-verify', 'login-{domain}',
        '{domain}-security', 'account-{domain}', '{domain}-signin', 'verify-{domain}',
        '{domain}-support', 'secure{domain}', '{domain}login', '{domain}update'
    ]
    
    urls = []
    labels = []
    
    # Generate legitimate URLs (50%)
    for _ in range(n_samples // 2):
        domain = np.random.choice(legitimate_domains)
        protocol = np.random.choice(['http', 'https'], p=[0.2, 0.8])
        
        # Add some path variations
        if np.random.random() < 0.6:
            paths = ['', '/home', '/about', '/contact', '/products', '/services', '/help']
            path = np.random.choice(paths)
        else:
            path = f'/page/{np.random.randint(1, 100)}'
        
        url = f"{protocol}://{domain}{path}"
        urls.append(url)
        labels.append(0)  # Legitimate
    
    # Generate phishing URLs (50%)
    for _ in range(n_samples // 2):
        base_domain = np.random.choice(legitimate_domains)
        pattern = np.random.choice(phishing_patterns)
        
        # Create suspicious domain
        if '{domain}' in pattern:
            suspicious_domain = pattern.format(domain=base_domain.replace('.', '-'))
        else:
            suspicious_domain = f"{pattern}-{base_domain.replace('.', '-')}"
        
        # Add suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.com', '.net', '.org']
        tld = np.random.choice(suspicious_tlds)
        
        # Sometimes use IP addresses
        if np.random.random() < 0.1:
            suspicious_domain = f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
        else:
            suspicious_domain += tld
        
        protocol = np.random.choice(['http', 'https'], p=[0.7, 0.3])  # Less HTTPS for phishing
        
        # Add suspicious paths
        suspicious_paths = [
            '/login', '/secure', '/update', '/verify', '/account', '/signin',
            '/security', '/confirm', '/suspended', '/urgent', '/alert'
        ]
        
        if np.random.random() < 0.8:
            path = np.random.choice(suspicious_paths)
        else:
            path = ''
        
        url = f"{protocol}://{suspicious_domain}{path}"
        urls.append(url)
        labels.append(1)  # Phishing
    
    # Create DataFrame
    df = pd.DataFrame({
        'url': urls,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save dataset
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/sample_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Sample dataset created: {output_path}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Legitimate: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    print(f"  - Phishing: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    
    return df

def extract_features_from_dataset(input_file, output_file=None, sample_size=None):
    """
    Extract features from a dataset of URLs.
    """
    from feature_extractor import FeatureExtractor
    
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    if 'url' not in df.columns:
        raise ValueError("Dataset must contain 'url' column")
    
    # Sample data if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size).reset_index(drop=True)
        print(f"Sampled {sample_size} URLs from dataset")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Extract features
    print("Extracting features...")
    features_list = []
    
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing {i+1}/{len(df)}...")
        
        url = row['url']
        try:
            features = extractor.extract_all_features(url)
            features['url'] = url
            
            # Add label if available
            if 'label' in row:
                features['label'] = row['label']
            
            features_list.append(features)
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    # Create features DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Save results
    if output_file is None:
        output_file = input_file.replace('.csv', '_features.csv')
    
    features_df.to_csv(output_file, index=False)
    
    print(f"✓ Features extracted and saved to {output_file}")
    print(f"  - Processed: {len(features_df)} URLs")
    print(f"  - Features: {len(features_df.columns) - 2} (excluding url and label)")  # -2 for url and label
    
    return features_df

def validate_dataset(dataset_path):
    """
    Validate a dataset for training.
    """
    print(f"Validating dataset: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    issues = []
    
    # Check required columns
    required_columns = ['url', 'label']
    for col in required_columns:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
    
    # Check label distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            issues.append("Dataset should have both phishing and legitimate samples")
        
        # Check for severe class imbalance
        min_class_ratio = min(label_counts) / len(df)
        if min_class_ratio < 0.1:
            issues.append(f"Severe class imbalance detected: {min_class_ratio:.1%} minority class")
    
    # Check URL format
    if 'url' in df.columns:
        invalid_urls = 0
        for url in df['url'].sample(min(100, len(df))):  # Check sample
            try:
                parsed = urlparse(str(url))
                if not parsed.scheme or not parsed.netloc:
                    invalid_urls += 1
            except:
                invalid_urls += 1
        
        if invalid_urls > 0:
            issues.append(f"Invalid URL format detected in {invalid_urls} samples")
    
    # Print results
    if issues:
        print("⚠️  Dataset validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Dataset validation passed")
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    
    if 'label' in df.columns:
        print(f"  - Label distribution:")
        for label, count in df['label'].value_counts().items():
            percentage = count / len(df) * 100
            label_name = 'Phishing' if label == 1 else 'Legitimate'
            print(f"    {label_name}: {count} ({percentage:.1f}%)")
    
    return len(issues) == 0

def split_dataset(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    print(f"Splitting dataset: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
    
    df = pd.read_csv(input_file)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split data
    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]
    
    # Save splits
    base_path = input_file.replace('.csv', '')
    train_path = f"{base_path}_train.csv"
    val_path = f"{base_path}_val.csv"
    test_path = f"{base_path}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Dataset split completed:")
    print(f"  - Train: {len(train_df)} samples → {train_path}")
    print(f"  - Validation: {len(val_df)} samples → {val_path}")
    print(f"  - Test: {len(test_df)} samples → {test_path}")
    
    return train_path, val_path, test_path

def benchmark_models(features_file):
    """
    Run a quick benchmark of all models.
    """
    from model_trainer import PhishingModelTrainer
    
    print("Running model benchmark...")
    
    # Initialize trainer
    trainer = PhishingModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data(features_file)
    
    # Quick training (no grid search)
    print("\nTraining models (fast mode)...")
    training_results = trainer.train_all_models(X_train, y_train, use_grid_search=True)
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluation_results = trainer.evaluate_all_models(X_test, y_test)
    
    # Print benchmark results
    print("\n" + "="*60)
    print("MODEL BENCHMARK RESULTS")
    print("="*60)
    
    results_table = []
    for model_name in trainer.models.keys():
        eval_result = evaluation_results.get(model_name, {})
        train_result = training_results.get(model_name, {})
        
        results_table.append([
            model_name,
            f"{eval_result.get('accuracy', 0):.3f}",
            f"{eval_result.get('precision', 0):.3f}",
            f"{eval_result.get('recall', 0):.3f}",
            f"{eval_result.get('f1_score', 0):.3f}",
            f"{eval_result.get('auc_score', 0):.3f}",
            f"{train_result.get('training_time', 0):.1f}s"
        ])
    
    # Print table
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Time"]
    col_widths = [max(len(str(row[i])) for row in [headers] + results_table) for i in range(len(headers))]
    
    # Print header
    header_row = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))
    
    # Print results
    for row in results_table:
        result_row = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
        print(result_row)
    
    print("="*60)
    
    # Find best model
    best_model = max(evaluation_results.keys(), 
                    key=lambda x: evaluation_results[x].get('f1_score', 0))
    print(f"Best performing model: {best_model}")

    # Saving the models
    trainer.save_models('models/')

    return evaluation_results

def generate_test_urls():
    """
    Generate a list of test URLs for demonstration.
    """
    test_urls = {
        'legitimate': [
            'https://www.google.com',
            'https://github.com/microsoft/vscode',
            'https://stackoverflow.com/questions/tagged/python',
            'https://www.amazon.com/dp/B08N5WRWNW',
            'https://en.wikipedia.org/wiki/Machine_learning',
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://www.linkedin.com/in/profile',
            'https://docs.python.org/3/tutorial/',
            'https://www.netflix.com/browse',
            'https://www.apple.com/iphone/'
        ],
        'suspicious': [
            'http://secure-paypal-update.tk/login',
            'https://amazon-security-check.ml/verify',
            'http://192.168.1.100/banking/login.php',
            'https://microsoft-account-suspended.ga/urgent',
            'http://google-verify-account.cf/signin',
            'https://apple-id-locked.net/unlock',
            'http://facebook-security-alert.org/confirm',
            'https://instagram-verify-account.tk/login',
            'http://twitter-account-suspended.ml/reactivate',
            'https://dropbox-storage-full.ga/upgrade'
        ]
    }
    
    # Create test file
    os.makedirs('data', exist_ok=True)
    
    all_urls = []
    labels = []
    
    for category, urls in test_urls.items():
        for url in urls:
            all_urls.append(url)
            labels.append(0 if category == 'legitimate' else 1)
    
    test_df = pd.DataFrame({
        'url': all_urls,
        'label': labels,
        'category': ['legitimate'] * len(test_urls['legitimate']) + ['suspicious'] * len(test_urls['suspicious'])
    })
    
    test_file = 'data/test_urls.csv'
    test_df.to_csv(test_file, index=False)
    
    print(f"✓ Generated test URLs: {test_file}")
    print(f"  - Legitimate: {len(test_urls['legitimate'])}")
    print(f"  - Suspicious: {len(test_urls['suspicious'])}")
    
    return test_file

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    print("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost',
        'requests', 'beautifulsoup4', 'flask',
        'lime', 'shap', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✓ All dependencies are installed")
        return True

def setup_project():
    """
    Setup the complete project structure and initial data.
    """
    print("Setting up Phishing Detection project...")
    
    # Create directory structure
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'logs',
        'notebooks',
        'tests',
        'web_app/templates',
        'web_app/static/css',
        'web_app/static/js',
        'scripts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Some dependencies are missing. Please install them first.")
        return False
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    create_sample_dataset(n_samples=2000)
    
    # Generate test URLs
    print("\nGenerating test URLs...")
    generate_test_urls()
    
    # Extract features from sample dataset
    print("\nExtracting features from sample dataset...")
    try:
        extract_features_from_dataset(
            'data/processed/sample_dataset.csv',
            'data/processed/sample_features.csv'
        )
    except Exception as e:
        print(f"⚠️  Feature extraction failed: {str(e)}")
        print("You may need to run this manually later.")
    
    # Create configuration file
    config = {
        'project_name': 'Phishing Detection with XAI',
        'version': '1.0.0',
        'created': datetime.now().isoformat(),
        'data_sources': {
            'sample_dataset': 'data/processed/sample_dataset.csv',
            'sample_features': 'data/processed/sample_features.csv',
            'test_urls': 'data/test_urls.csv'
        },
        'model_config': {
            'default_model': 'xgboost',
            'available_models': ['random_forest', 'xgboost', 'neural_network']
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created configuration file: config.json")
    
    print("\n" + "="*50)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Train models: python src/model_trainer.py")
    print("2. Run web app: python web_app/app.py")
    print("3. Test prediction: python src/predictor.py --url https://example.com")
    print("\nFor more information, see README.md")
    
    return True

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Phishing Detection Utilities')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project structure')
    
    # Download data command
    download_parser = subparsers.add_parser('download-data', help='Download datasets')
    
    # Create sample command
    sample_parser = subparsers.add_parser('create-sample', help='Create sample dataset')
    sample_parser.add_argument('--size', type=int, default=5000, help='Number of samples')
    
    # Extract features command
    features_parser = subparsers.add_parser('extract-features', help='Extract features from URLs')
    features_parser.add_argument('input_file', help='Input CSV file with URLs')
    features_parser.add_argument('--output', help='Output file for features')
    features_parser.add_argument('--sample-size', type=int, help='Sample size to process')
    
    # Validate dataset command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('dataset_file', help='Dataset file to validate')
    
    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset')
    split_parser.add_argument('input_file', help='Input dataset file')
    split_parser.add_argument('--train', type=float, default=0.7, help='Train ratio')
    split_parser.add_argument('--val', type=float, default=0.15, help='Validation ratio')
    split_parser.add_argument('--test', type=float, default=0.15, help='Test ratio')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark models')
    benchmark_parser.add_argument('features_file', help='Features file for benchmarking')
    
    # Generate test URLs command
    test_urls_parser = subparsers.add_parser('generate-test-urls', help='Generate test URLs')
    
    # Check dependencies command
    deps_parser = subparsers.add_parser('check-deps', help='Check dependencies')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'setup':
            setup_project()
        
        elif args.command == 'download-data':
            download_phishing_datasets()
        
        elif args.command == 'create-sample':
            create_sample_dataset(args.size)
        
        elif args.command == 'extract-features':
            extract_features_from_dataset(args.input_file, args.output, args.sample_size)
        
        elif args.command == 'validate':
            validate_dataset(args.dataset_file)
        
        elif args.command == 'split':
            split_dataset(args.input_file, args.train, args.val, args.test)
        
        elif args.command == 'benchmark':
            benchmark_models(args.features_file)
        
        elif args.command == 'generate-test-urls':
            generate_test_urls()
        
        elif args.command == 'check-deps':
            check_dependencies()
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())