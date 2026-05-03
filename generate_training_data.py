"""
Generate Realistic Training Data for Phishing Detection
Creates feature vectors with proper distributions to ensure the model
learns meaningful patterns across ALL features.
"""

import pandas as pd
import numpy as np
import os

def generate_training_data(n_samples=10000, seed=42):
    """
    Generate realistic training data with well-separated feature distributions
    for phishing vs legitimate URLs.
    """
    np.random.seed(seed)
    
    n_legit = n_samples // 2
    n_phish = n_samples - n_legit
    
    print(f"Generating {n_legit} legitimate + {n_phish} phishing samples...")
    
    # =============================================
    # LEGITIMATE SITE FEATURES
    # =============================================
    legit = {}
    
    # URL structure - legitimate sites have shorter, cleaner URLs
    legit['url_length'] = np.random.normal(45, 15, n_legit).clip(15, 120).astype(int)
    legit['domain_length'] = np.random.normal(12, 5, n_legit).clip(4, 30).astype(int)
    legit['path_length'] = np.random.normal(15, 10, n_legit).clip(0, 60).astype(int)
    
    # Character analysis - legitimate sites are clean
    legit['special_char_count'] = np.random.poisson(2, n_legit).clip(0, 10)
    legit['dots_count'] = np.random.choice([2, 3, 2, 2, 3, 4], n_legit)
    legit['hyphens_count'] = np.random.choice([0, 0, 0, 1, 1, 2], n_legit)
    legit['underscores_count'] = np.random.choice([0, 0, 0, 0, 1], n_legit)
    legit['digits_count'] = np.random.poisson(1, n_legit).clip(0, 8)
    
    # Protocol - legitimate sites use HTTPS
    legit['has_https'] = np.random.choice([0, 1], n_legit, p=[0.1, 0.9])
    legit['has_www'] = np.random.choice([0, 1], n_legit, p=[0.4, 0.6])
    legit['has_at_symbol'] = np.zeros(n_legit, dtype=int)
    legit['has_ip_address'] = np.zeros(n_legit, dtype=int)
    
    # URL structure
    legit['subdomain_count'] = np.random.choice([0, 0, 0, 1, 1, 2], n_legit)
    legit['url_depth'] = np.random.choice([0, 1, 1, 2, 2, 3], n_legit)
    legit['has_port'] = np.zeros(n_legit, dtype=int)
    
    # Query parameters
    legit['query_length'] = np.random.exponential(5, n_legit).clip(0, 50).astype(int)
    legit['params_count'] = np.random.choice([0, 0, 0, 1, 1, 2, 3], n_legit)
    
    # Suspicious patterns - legitimate sites rarely have these
    legit['has_suspicious_words'] = np.random.choice([0, 0, 0, 0, 0, 1], n_legit)
    legit['entropy'] = np.random.normal(3.2, 0.4, n_legit).clip(2.0, 4.5)
    
    # Content features - legitimate sites are rich
    legit['has_login_form'] = np.random.choice([0, 0, 0, 0, 1], n_legit)
    legit['forms_count'] = np.random.choice([0, 1, 1, 2, 2, 3], n_legit)
    legit['input_fields_count'] = np.random.poisson(3, n_legit).clip(0, 15)
    legit['has_javascript'] = np.random.choice([0, 1], n_legit, p=[0.05, 0.95])
    legit['scripts_count'] = np.random.poisson(8, n_legit).clip(0, 30)
    legit['has_iframes'] = np.random.choice([0, 0, 0, 1], n_legit)
    legit['iframes_count'] = np.random.choice([0, 0, 0, 0, 1, 2], n_legit)
    legit['total_links'] = np.random.poisson(25, n_legit).clip(2, 200)
    legit['external_links'] = np.random.poisson(5, n_legit).clip(0, 50)
    legit['external_links_ratio'] = (legit['external_links'] / np.maximum(legit['total_links'], 1)).clip(0, 1)
    legit['images_count'] = np.random.poisson(8, n_legit).clip(0, 100)
    legit['has_favicon'] = np.random.choice([0, 1], n_legit, p=[0.05, 0.95])
    legit['content_length'] = np.random.lognormal(10, 1, n_legit).clip(1000, 500000).astype(int)
    legit['has_title'] = np.ones(n_legit, dtype=int)
    legit['meta_tags_count'] = np.random.poisson(8, n_legit).clip(2, 25)
    
    # Suspicious content patterns - rarely present in legit
    legit['has_meta_refresh'] = np.zeros(n_legit, dtype=int)
    legit['has_popup_patterns'] = np.random.choice([0, 0, 0, 0, 1], n_legit)
    legit['has_suspicious_js'] = np.random.choice([0, 0, 0, 0, 1, 2], n_legit)
    
    # Host features - legitimate sites are well-established
    legit['domain_age'] = np.random.uniform(365, 7300, n_legit).astype(int)  # 1-20 years
    legit['domain_expiry_days'] = np.random.uniform(180, 3650, n_legit).astype(int)
    legit['has_ssl'] = np.random.choice([0, 1], n_legit, p=[0.05, 0.95])
    # Allow small ssl age. Modern setups like Let's Encrypt rotate every ~30-60 days
    legit['ssl_age'] = np.random.uniform(1, 730, n_legit).astype(int)
    legit['ssl_valid'] = legit['has_ssl'].copy()
    legit['dns_records_count'] = np.random.poisson(8, n_legit).clip(2, 25)
    legit['has_mx_record'] = np.random.choice([0, 1], n_legit, p=[0.1, 0.9])
    legit['has_spf_record'] = np.random.choice([0, 1], n_legit, p=[0.15, 0.85])
    legit['has_dmarc_record'] = np.random.choice([0, 1], n_legit, p=[0.25, 0.75])
    
    # =============================================
    # PHISHING SITE FEATURES
    # =============================================
    phish = {}
    
    # URL structure - phishing sites have longer, messier URLs
    phish['url_length'] = np.random.normal(85, 30, n_phish).clip(30, 250).astype(int)
    phish['domain_length'] = np.random.normal(22, 8, n_phish).clip(5, 50).astype(int)
    phish['path_length'] = np.random.normal(35, 20, n_phish).clip(0, 120).astype(int)
    
    # Character analysis - phishing sites are messy
    phish['special_char_count'] = np.random.poisson(6, n_phish).clip(1, 25)
    phish['dots_count'] = np.random.choice([3, 4, 4, 5, 5, 6, 7], n_phish)
    phish['hyphens_count'] = np.random.choice([0, 1, 2, 2, 3, 3, 4], n_phish)
    phish['underscores_count'] = np.random.choice([0, 0, 1, 1, 2, 3], n_phish)
    phish['digits_count'] = np.random.poisson(4, n_phish).clip(0, 15)
    
    # Protocol - phishing sites often lack HTTPS
    phish['has_https'] = np.random.choice([0, 1], n_phish, p=[0.45, 0.55])
    phish['has_www'] = np.random.choice([0, 1], n_phish, p=[0.7, 0.3])
    phish['has_at_symbol'] = np.random.choice([0, 1], n_phish, p=[0.9, 0.1])
    phish['has_ip_address'] = np.random.choice([0, 1], n_phish, p=[0.85, 0.15])
    
    # URL structure - more complex
    phish['subdomain_count'] = np.random.choice([1, 2, 2, 3, 3, 4, 5], n_phish)
    phish['url_depth'] = np.random.choice([2, 3, 3, 4, 4, 5], n_phish)
    phish['has_port'] = np.random.choice([0, 0, 0, 0, 1], n_phish)
    
    # Query parameters - often used for tracking/obfuscation
    phish['query_length'] = np.random.exponential(15, n_phish).clip(0, 100).astype(int)
    phish['params_count'] = np.random.choice([0, 1, 2, 2, 3, 4, 5], n_phish)
    
    # Suspicious patterns - common in phishing
    phish['has_suspicious_words'] = np.random.choice([0, 1, 1, 2, 2, 3, 4], n_phish)
    phish['entropy'] = np.random.normal(4.0, 0.5, n_phish).clip(2.5, 5.5)
    
    # Content features - phishing sites are sparse but targeted
    phish['has_login_form'] = np.random.choice([0, 1], n_phish, p=[0.3, 0.7])
    phish['forms_count'] = np.random.choice([0, 1, 1, 1, 2], n_phish)
    phish['input_fields_count'] = np.random.poisson(4, n_phish).clip(0, 12)
    phish['has_javascript'] = np.random.choice([0, 1], n_phish, p=[0.15, 0.85])
    phish['scripts_count'] = np.random.poisson(5, n_phish).clip(0, 20)
    phish['has_iframes'] = np.random.choice([0, 1], n_phish, p=[0.6, 0.4])
    phish['iframes_count'] = np.random.choice([0, 0, 1, 1, 2, 3], n_phish)
    phish['total_links'] = np.random.poisson(10, n_phish).clip(0, 50)
    phish['external_links'] = np.random.poisson(5, n_phish).clip(0, 30)
    phish['external_links_ratio'] = (phish['external_links'] / np.maximum(phish['total_links'], 1)).clip(0, 1)
    phish['images_count'] = np.random.poisson(5, n_phish).clip(0, 30)
    phish['has_favicon'] = np.random.choice([0, 1], n_phish, p=[0.5, 0.5])
    phish['content_length'] = np.random.lognormal(8, 1, n_phish).clip(200, 100000).astype(int)
    phish['has_title'] = np.random.choice([0, 1], n_phish, p=[0.2, 0.8])
    phish['meta_tags_count'] = np.random.poisson(3, n_phish).clip(0, 10)
    
    # Suspicious content patterns - common in phishing
    phish['has_meta_refresh'] = np.random.choice([0, 1], n_phish, p=[0.7, 0.3])
    phish['has_popup_patterns'] = np.random.poisson(2, n_phish).clip(0, 8)
    phish['has_suspicious_js'] = np.random.poisson(2, n_phish).clip(0, 10)
    
    # Host features - phishing sites are new and poorly configured
    phish['domain_age'] = np.random.exponential(20, n_phish).clip(0, 180).astype(int)
    phish['domain_expiry_days'] = np.random.uniform(10, 365, n_phish).astype(int)
    phish['has_ssl'] = np.random.choice([0, 1], n_phish, p=[0.35, 0.65])
    phish['ssl_age'] = np.where(
        phish['has_ssl'] == 1,
        np.random.uniform(1, 60, n_phish).astype(int),
        -1
    )
    phish['ssl_valid'] = np.where(
        phish['has_ssl'] == 1,
        np.random.choice([0, 1], n_phish, p=[0.3, 0.7]),
        0
    )
    phish['dns_records_count'] = np.random.poisson(2, n_phish).clip(0, 8)
    phish['has_mx_record'] = np.random.choice([0, 1], n_phish, p=[0.7, 0.3])
    phish['has_spf_record'] = np.random.choice([0, 1], n_phish, p=[0.8, 0.2])
    phish['has_dmarc_record'] = np.random.choice([0, 1], n_phish, p=[0.9, 0.1])
    
    # =============================================
    # COMBINE INTO DATAFRAME
    # =============================================
    legit_df = pd.DataFrame(legit)
    legit_df['label'] = 0
    
    phish_df = pd.DataFrame(phish)
    phish_df['label'] = 1
    
    # Add some noise/overlap to make it realistic
    # Some legit sites might look somewhat suspicious (e.g., new startups)
    noise_idx = np.random.choice(n_legit, size=int(n_legit * 0.05), replace=False)
    legit_df.loc[noise_idx, 'domain_age'] = np.random.uniform(10, 90, len(noise_idx)).astype(int)
    legit_df.loc[noise_idx, 'has_ssl'] = np.random.choice([0, 1], len(noise_idx), p=[0.3, 0.7])
    
    # Some phishing sites use HTTPS and have valid SSL (modern phishing)
    modern_idx = np.random.choice(n_phish, size=int(n_phish * 0.15), replace=False)
    phish_df.loc[modern_idx, 'has_https'] = 1
    phish_df.loc[modern_idx, 'has_ssl'] = 1
    phish_df.loc[modern_idx, 'ssl_valid'] = 1
    
    # Combine and shuffle
    df = pd.concat([legit_df, phish_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save
    os.makedirs('data', exist_ok=True)
    output_path = 'data/dataset_features_synthetic.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated dataset saved to {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Legitimate: {(df['label'] == 0).sum()}")
    print(f"   Phishing: {(df['label'] == 1).sum()}")
    
    # Print feature distribution comparison
    print("\n📊 Feature Distribution Comparison (mean values):")
    print(f"{'Feature':<25} {'Legit':>10} {'Phishing':>10} {'Diff':>10}")
    print("-" * 55)
    for col in df.columns:
        if col == 'label':
            continue
        legit_mean = df[df['label'] == 0][col].mean()
        phish_mean = df[df['label'] == 1][col].mean()
        diff = abs(phish_mean - legit_mean)
        marker = " ⚠️" if diff < 0.1 else ""
        print(f"{col:<25} {legit_mean:>10.2f} {phish_mean:>10.2f} {diff:>10.2f}{marker}")
    
    return df


if __name__ == "__main__":
    generate_training_data()
