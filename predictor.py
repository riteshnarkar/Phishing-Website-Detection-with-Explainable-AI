"""
Prediction Pipeline for Phishing Detection
Combines feature extraction, model prediction, and explanation generation.
"""

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import warnings
import re
from urllib.parse import urlparse
warnings.filterwarnings('ignore')

from feature_extractor import FeatureExtractor
from explainer import PhishingExplainer

class PhishingPredictor:
    """
    Complete prediction pipeline for phishing detection.
    Handles feature extraction, prediction, and explanation generation.
    """
    
    def __init__(self, models_dir='models/', default_model='xgboost'):
        """
        Initialize the prediction pipeline.
        
        Args:
            models_dir (str): Directory containing trained models
            default_model (str): Default model to use for predictions
        """
        self.models_dir = models_dir
        self.default_model = default_model
        self.models = {}
        self.scalers = {}
        self.explainers = {}
        self.feature_names = []
        self.feature_extractor = FeatureExtractor()
        
        # Load models and setup
        self._load_models()
        self._setup_explainers()
    
    def _load_models(self):
        """Load trained models and metadata."""
        import os
        
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory {self.models_dir} not found")
        
        # Load metadata
        metadata_path = os.path.join(self.models_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        
        # Load scalers
        scalers_path = os.path.join(self.models_dir, 'scalers.joblib')
        self.scalers = joblib.load(scalers_path)
        
        # Load models
        model_files = {
            'random_forest': 'random_forest_model.joblib',
            'xgboost': 'xgboost_model.joblib',
            'neural_network': 'neural_network_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model")
        
        if not self.models:
            raise ValueError("No models found in the specified directory")
        
        # Ensure default model exists
        if self.default_model not in self.models:
            self.default_model = list(self.models.keys())[0]
            print(f"Default model not found, using {self.default_model}")
    
    def _setup_explainers(self):
        """Setup explainers for all loaded models."""
        for model_name, model in self.models.items():
            scaler = self.scalers.get(model_name)
            self.explainers[model_name] = PhishingExplainer(
                model, scaler, self.feature_names
            )
    
    def extract_features(self, url):
        """
        Extract features from a URL.
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: Extracted features
        """
        try:
            features = self.feature_extractor.extract_all_features(url)
            
            # Ensure all expected features are present
            for feature_name in self.feature_names:
                if feature_name not in features:
                    features[feature_name] = -1  # Default value for missing features
            
            return features
        except Exception as e:
            print(f"Error extracting features from {url}: {str(e)}")
            # Return default features if extraction fails
            return {name: -1 for name in self.feature_names}
    
    def predict_url(self, url, model_name=None, include_explanation=True, explanation_level='basic'):
        """
        Predict whether a URL is phishing or legitimate.
        
        Args:
            url (str): URL to analyze
            model_name (str): Model to use (default: self.default_model)
            include_explanation (bool): Whether to include explanation
            explanation_level (str): Level of explanation ('basic', 'detailed', 'comprehensive')
            
        Returns:
            dict: Prediction results with optional explanation
        """
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available")
        
        # Extract features
        print(f"Analyzing URL: {url}")
        features_dict = self.extract_features(url)
        
        # Convert to array in correct order
        features_array = np.array([features_dict[name] for name in self.feature_names])
        features_array = features_array.reshape(1, -1)
        
        # Get model and scaler
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Scale features if needed
        if scaler:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get raw model probability
        raw_phishing_prob = float(prediction_proba[1])
        
        # Apply heuristic guardrails to catch false negatives
        phishing_prob, guardrail_reasons = self._apply_heuristic_guardrails(
            raw_phishing_prob, features_dict, url
        )
        
        if guardrail_reasons:
            print(f"GUARDRAIL: Adjusted phishing_prob {raw_phishing_prob:.4f} -> {phishing_prob:.4f}")
            for reason in guardrail_reasons:
                print(f"  - {reason}")
        
        # Use adjusted threshold (0.45 instead of 0.5) for higher phishing sensitivity
        PHISHING_THRESHOLD = 0.45
        prediction = 1 if phishing_prob >= PHISHING_THRESHOLD else 0
        
        # Calculate confidence and risk level
        confidence = phishing_prob if prediction == 1 else (1 - phishing_prob)
        
        result = {
            'url': url,
            'prediction': 'phishing' if prediction == 1 else 'legitimate',
            'confidence': float(confidence),
            'phishing_probability': float(phishing_prob),
            'raw_model_probability': raw_phishing_prob,
            'risk_level': self._get_risk_level(phishing_prob, prediction),
            'model_used': model_name,
            'timestamp': datetime.now().isoformat(),
            'features': features_dict,
            'guardrail_adjustments': guardrail_reasons
        }
        
        # CORRECTED: Use enhanced explainer for all explanation levels
        if include_explanation:
            try:
                # Always use the enhanced explainer - no internal methods
                explainer = self.explainers[model_name]
                
                # Debug logging
                print(f"DEBUG: Using explanation level: {explanation_level}")
                print(f"DEBUG: Explainer type: {type(explainer)}")
                
                # Call the enhanced explainer with the explanation level
                explanation_report = explainer.create_explanation_report(
                    features_array, url, explanation_level
                )
                
                print(f"DEBUG: Explanation report keys: {list(explanation_report.keys())}")
                
                # Check if we got detailed explanation
                if 'detailed_explanation' in explanation_report and explanation_report['detailed_explanation']:
                    result['detailed_explanation'] = explanation_report['detailed_explanation']
                    result['explanation'] = explanation_report['explanation_text']
                    print(f"DEBUG: Using detailed explanation with {len(explanation_report['detailed_explanation']['sections'])} sections")
                else:
                    # Basic explanation only
                    result['explanation'] = explanation_report['explanation_text']
                    print(f"DEBUG: Using basic explanation only")
                    
            except Exception as e:
                print(f"WARNING: Could not generate explanation: {str(e)}")
                import traceback
                print(f"TRACEBACK: {traceback.format_exc()}")
                result['explanation'] = f"Analysis complete. URL classified as {result['prediction']} with {confidence:.1%} confidence."
        
        return result
    
    def predict_batch(self, urls, model_name=None, include_explanations=False, explanation_level='basic'):
        """
        Predict multiple URLs.
        
        Args:
            urls (list): List of URLs to analyze
            model_name (str): Model to use
            include_explanations (bool): Whether to include explanations
            explanation_level (str): Level of explanation ('basic', 'detailed', 'comprehensive')
            
        Returns:
            list: List of prediction results
        """
        results = []
        total_urls = len(urls)
        
        print(f"Analyzing {total_urls} URLs...")
        
        for i, url in enumerate(urls):
            try:
                print(f"Processing {i+1}/{total_urls}: {url}")
                result = self.predict_url(url, model_name, include_explanations, explanation_level)
                results.append(result)
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                results.append({
                    'url': url,
                    'prediction': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def _apply_heuristic_guardrails(self, raw_phishing_prob, features_dict, url):
        """
        Apply heuristic guardrails to catch obvious phishing signals
        that the ML model might miss (reducing false negatives), and
        discount probabilities for extremely trusted domains (reducing false positives).
        
        Args:
            raw_phishing_prob (float): Raw model prediction probability
            features_dict (dict): Extracted features
            url (str): The URL being analyzed
            
        Returns:
            tuple: (adjusted_probability, list_of_reasons)
        """
        adjustments = []
        boost = 0.0
        
        # 1. IP address as domain — very strong phishing signal
        if features_dict.get('has_ip_address', 0) == 1:
            boost += 0.30
            adjustments.append("IP address used as domain (+0.30)")
        
        # 2. @ symbol in URL — URL obfuscation technique
        if features_dict.get('has_at_symbol', 0) == 1:
            boost += 0.20
            adjustments.append("@ symbol in URL - potential redirect trick (+0.20)")
        
        # 3. Very new domain (< 30 days) AND domain age was successfully retrieved
        domain_age = features_dict.get('domain_age', -1)
        if 0 <= domain_age < 30:
            boost += 0.15
            adjustments.append(f"Very new domain ({domain_age} days old) (+0.15)")
        elif 30 <= domain_age < 90:
            boost += 0.08
            adjustments.append(f"New domain ({domain_age} days old) (+0.08)")
        
        # 4. No HTTPS + has login form — credential harvesting risk
        if features_dict.get('has_https', 0) == 0 and features_dict.get('has_login_form', 0) == 1:
            boost += 0.20
            adjustments.append("Login form on non-HTTPS page (+0.20)")
        
        # 5. No HTTPS alone is a moderate signal
        elif features_dict.get('has_https', 0) == 0:
            boost += 0.05
            adjustments.append("No HTTPS encryption (+0.05)")
        
        # 6. Extremely long URL with many special characters
        url_length = features_dict.get('url_length', 0)
        special_chars = features_dict.get('special_char_count', 0)
        if url_length > 100 and special_chars > 8:
            boost += 0.10
            adjustments.append(f"Long URL ({url_length} chars) with many special chars ({special_chars}) (+0.10)")
        elif url_length > 150:
            boost += 0.10
            adjustments.append(f"Extremely long URL ({url_length} chars) (+0.10)")
        
        # 7. Multiple suspicious keywords (≥2)
        suspicious_words = features_dict.get('has_suspicious_words', 0)
        if suspicious_words >= 3:
            boost += 0.12
            adjustments.append(f"Multiple suspicious keywords ({suspicious_words} found) (+0.12)")
        elif suspicious_words >= 2:
            boost += 0.06
            adjustments.append(f"Suspicious keywords ({suspicious_words} found) (+0.06)")
        
        # 8. Has suspicious JavaScript patterns
        if features_dict.get('has_suspicious_js', 0) >= 2:
            boost += 0.08
            adjustments.append(f"Suspicious JavaScript patterns detected (+0.08)")
        
        # 9. Meta refresh redirect (auto-redirect)
        if features_dict.get('has_meta_refresh', 0) == 1:
            boost += 0.06
            adjustments.append("Auto-redirect via meta refresh (+0.06)")
        
        # 10. No SSL + no MX record + no SPF (poorly configured domain)
        if (features_dict.get('has_ssl', 0) == 0 and 
            features_dict.get('has_mx_record', 0) == 0 and 
            features_dict.get('has_spf_record', 0) == 0):
            boost += 0.10
            adjustments.append("Poorly configured domain (no SSL, MX, or SPF) (+0.10)")
            
        # POSITIVE GUARDRAILS (Discounts for highly legitimate signs)
        discount = 0.0
        
        # 11. Ancient domain (> 10 years / 3650 days) with valid SSL + MX
        domain_age = features_dict.get('domain_age', -1)
        if domain_age > 3650 and features_dict.get('has_ssl', 0) == 1 and features_dict.get('has_mx_record', 0) == 1:
            discount += 0.60
            adjustments.append(f"Highly established ancient domain (>10 years) with valid SSL/MX (-0.60)")
            
        # 12. Very old established domain (> 3 years / 1000 days) with valid SSL + MX
        elif domain_age > 1000 and features_dict.get('has_ssl', 0) == 1 and features_dict.get('has_mx_record', 0) == 1:
            discount += 0.30
            adjustments.append(f"Established older domain (>3 years) with valid SSL/MX (-0.30)")

        # 13. Perfect email/DNS configurations (MX + SPF + DMARC)
        if (features_dict.get('has_mx_record', 0) == 1 and 
            features_dict.get('has_spf_record', 0) == 1 and 
            features_dict.get('has_dmarc_record', 0) == 1):
            discount += 0.15
            adjustments.append("Perfectly configured DNS/Email framework (MX, SPF, DMARC) (-0.15)")
        
        # Apply boost with diminishing returns (cap at 0.98 total probability)
        adjusted_prob = raw_phishing_prob
        
        if boost > 0:
            adjusted_prob = adjusted_prob + boost * (1 - adjusted_prob)
            adjusted_prob = min(adjusted_prob, 0.98)  # Cap at 0.98
            
        # Apply discounts 
        if discount > 0:
            # Multiplicative reduction
            adjusted_prob = adjusted_prob * (1 - discount)
            # Ensure it doesn't go below 0.001
            adjusted_prob = max(adjusted_prob, 0.001)
        
        return adjusted_prob, adjustments
    
    def _get_risk_level(self, probability, prediction):
        """Determine risk level based on prediction probability."""
        confidence = probability if prediction == 1 else (1 - probability)
        
        if confidence > 0.9:
            return "very_high" if prediction == 1 else "very_low"
        elif confidence > 0.7:
            return "high" if prediction == 1 else "low"
        else:
            return "medium"
    
    def get_model_info(self):
        """
        Get information about loaded models.
        
        Returns:
            dict: Model information
        """
        return {
            'available_models': list(self.models.keys()),
            'default_model': self.default_model,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'explanation_levels': ['basic', 'detailed', 'comprehensive']
        }
    
    def save_results(self, results, filename):
        """
        Save prediction results to file.
        
        Args:
            results (list): Prediction results
            filename (str): Output filename
        """
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
        elif filename.endswith('.csv'):
            # Convert to DataFrame for CSV export
            df_data = []
            for result in results:
                row = {
                    'url': result.get('url', ''),
                    'prediction': result.get('prediction', ''),
                    'confidence': result.get('confidence', 0),
                    'phishing_probability': result.get('phishing_probability', 0),
                    'risk_level': result.get('risk_level', ''),
                    'model_used': result.get('model_used', ''),
                    'timestamp': result.get('timestamp', '')
                }
                
                # Add basic explanation if available
                if 'explanation' in result and isinstance(result['explanation'], str):
                    row['explanation'] = result['explanation']
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(filename, index=False)
        else:
            raise ValueError("Filename must end with .json or .csv")
        
        print(f"Results saved to {filename}")
    
    def analyze_url_comprehensive(self, url, model_name=None):
        """
        Perform comprehensive analysis with all available details.
        
        Args:
            url (str): URL to analyze
            model_name (str): Model to use
            
        Returns:
            dict: Comprehensive analysis results
        """
        # Get prediction with comprehensive explanation
        result = self.predict_url(url, model_name, True, 'comprehensive')
        
        # Add additional analysis
        features_dict = result['features']
        
        # Statistical analysis
        result['statistical_analysis'] = {
            'feature_statistics': self._calculate_feature_statistics(features_dict),
            'anomaly_detection': self._detect_anomalies(features_dict),
            'similarity_analysis': self._analyze_similarity_to_known_patterns(features_dict)
        }
        
        return result
    
    def _calculate_feature_statistics(self, features_dict):
        """Calculate statistical properties of extracted features."""
        stats = {
            'total_features': len(features_dict),
            'numeric_features': sum(1 for v in features_dict.values() if isinstance(v, (int, float))),
            'binary_features': sum(1 for v in features_dict.values() if v in [0, 1]),
            'missing_features': sum(1 for v in features_dict.values() if v == -1)
        }
        
        numeric_values = [v for v in features_dict.values() if isinstance(v, (int, float)) and v != -1]
        if numeric_values:
            stats['mean_value'] = np.mean(numeric_values)
            stats['std_deviation'] = np.std(numeric_values)
            stats['max_value'] = np.max(numeric_values)
            stats['min_value'] = np.min(numeric_values)
        
        return stats
    
    def _detect_anomalies(self, features_dict):
        """Detect anomalous feature values."""
        anomalies = []
        
        # Check for extreme values
        url_length = features_dict.get('url_length', 0)
        if url_length > 200:
            anomalies.append(f"Extremely long URL ({url_length} characters) - 99th percentile")
        
        num_dots = features_dict.get('num_dots', 0)
        if num_dots > 6:
            anomalies.append(f"Excessive dots in URL ({num_dots}) - potential domain confusion")
        
        special_chars = features_dict.get('num_special_chars', 0)
        if special_chars > 15:
            anomalies.append(f"High special character count ({special_chars}) - potential obfuscation")
        
        return anomalies if anomalies else ["No significant anomalies detected"]
    
    def _analyze_similarity_to_known_patterns(self, features_dict):
        """Analyze similarity to known phishing/legitimate patterns."""
        patterns = []
        
        # Common phishing patterns
        if (features_dict.get('url_length', 0) > 80 and 
            features_dict.get('num_dots', 0) > 3 and
            features_dict.get('num_special_chars', 0) > 5):
            patterns.append("Matches common phishing pattern: Long URL + Multiple subdomains + Special characters")
        
        # Legitimate patterns
        if (features_dict.get('url_length', 0) < 50 and 
            features_dict.get('https', 0) == 1 and
            features_dict.get('num_dots', 0) <= 2):
            patterns.append("Matches legitimate pattern: Short URL + HTTPS + Simple domain structure")
        
        return patterns if patterns else ["No specific pattern matches identified"]


def load_url_list(filename):
    """
    Load URLs from file.
    
    Args:
        filename (str): Path to file containing URLs (one per line)
        
    Returns:
        list: List of URLs
    """
    urls = []
    with open(filename, 'r') as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith('#'):
                urls.append(url)
    return urls


def main():
    """Main function for command-line usage with enhanced options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Phishing URL Detection')
    parser.add_argument('--url', type=str, help='Single URL to analyze')
    parser.add_argument('--urls-file', type=str, help='File containing URLs to analyze')
    parser.add_argument('--model', type=str, default='xgboost', 
                       help='Model to use (default: xgboost)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--explain', action='store_true', help='Include explanations')
    parser.add_argument('--explanation-level', type=str, default='basic',
                       choices=['basic', 'detailed', 'comprehensive'],
                       help='Level of explanation detail (default: basic)')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Perform comprehensive analysis with all details')
    parser.add_argument('--models-dir', type=str, default='models/', 
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = PhishingPredictor(models_dir=args.models_dir, default_model=args.model)
        print(f"Predictor initialized successfully with {len(predictor.models)} models")
    except Exception as e:
        print(f"Error initializing predictor: {str(e)}")
        return
    
    # Process URLs
    results = []
    
    if args.url:
        # Single URL analysis
        try:
            if args.comprehensive:
                result = predictor.analyze_url_comprehensive(args.url, args.model)
                print("\n" + "="*80)
                print("COMPREHENSIVE URL ANALYSIS")
                print("="*80)
            else:
                result = predictor.predict_url(args.url, args.model, args.explain, args.explanation_level)
                print("\n" + "="*60)
                print("URL ANALYSIS RESULTS")
                print("="*60)
            
            results.append(result)
            
            # Display basic results
            print(f"URL: {result['url']}")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Risk Level: {result['risk_level'].replace('_', ' ').title()}")
            print(f"Model Used: {result['model_used']}")
            
            # Display explanations based on level
            if args.explain or args.comprehensive:
                print("\n" + "-"*60)
                print("DETAILED EXPLANATION")
                print("-"*60)
                
                if 'detailed_explanation' in result:
                    print(f"{result['detailed_explanation']['summary']}\n")
                    
                    for section in result['detailed_explanation']['sections']:
                        print(f"{section['title']}")
                        print("-" * len(section['title']))
                        for item in section['content']:
                            if item.strip():  # Skip empty lines
                                print(f"  {item}")
                        print()
                        
                elif 'explanation' in result:
                    print(f"{result['explanation']}\n")
            
            # Display comprehensive analysis if requested
            if args.comprehensive and 'statistical_analysis' in result:
                print("\n" + "-"*60)
                print("STATISTICAL ANALYSIS")
                print("-"*60)
                stats = result['statistical_analysis']
                
                print("Feature Statistics:")
                for key, value in stats['feature_statistics'].items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
                
                print("\nAnomaly Detection:")
                for anomaly in stats['anomaly_detection']:
                    print(f"   • {anomaly}")
                
                print("\nPattern Analysis:")
                for pattern in stats['similarity_analysis']:
                    print(f"   • {pattern}")
                    
        except Exception as e:
            print(f"Error analyzing URL: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return
    
    elif args.urls_file:
        # Multiple URLs from file
        try:
            urls = load_url_list(args.urls_file)
            print(f"Loaded {len(urls)} URLs from {args.urls_file}")
            
            results = predictor.predict_batch(urls, args.model, args.explain, args.explanation_level)
            
            # Print summary
            total = len(results)
            phishing_count = sum(1 for r in results if r.get('prediction') == 'phishing')
            legitimate_count = sum(1 for r in results if r.get('prediction') == 'legitimate')
            error_count = sum(1 for r in results if r.get('prediction') == 'error')
            
            print("\n" + "="*60)
            print("BATCH ANALYSIS SUMMARY")
            print("="*60)
            print(f"Total URLs Processed: {total}")
            print(f"Phishing URLs: {phishing_count} ({phishing_count/total:.1%})")
            print(f"Legitimate URLs: {legitimate_count} ({legitimate_count/total:.1%})")
            print(f"Analysis Errors: {error_count} ({error_count/total:.1%})")
            
            # Show individual results
            print(f"\nIndividual Results:")
            for i, result in enumerate(results[:10]):  # Show first 10
                status_icon = "PHISHING" if result.get('prediction') == 'phishing' else "LEGITIMATE" if result.get('prediction') == 'legitimate' else "ERROR"
                confidence = result.get('confidence', 0)
                print(f"   {i+1:2d}. {status_icon} {result.get('url', '')[:50]}{'...' if len(result.get('url', '')) > 50 else ''} ({confidence:.1%})")
            
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more results")
            
        except FileNotFoundError:
            print(f"Error: File {args.urls_file} not found")
            return
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return
    
    else:
        print("Please provide either --url or --urls-file")
        parser.print_help()
        return
    
    # Save results if output file specified
    if args.output and results:
        try:
            predictor.save_results(results, args.output)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    print(f"\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()