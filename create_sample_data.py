import pandas as pd
import numpy as np
import os

def create_sample_dataset(n_samples=5000):
    """Create a realistic sample dataset for training."""
    np.random.seed(42)
    
    # Legitimate domains
    legitimate_domains = [
        'google.com', 'amazon.com', 'facebook.com', 'microsoft.com', 'apple.com',
        'youtube.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'github.com',
        'stackoverflow.com', 'wikipedia.org', 'reddit.com', 'netflix.com', 'paypal.com'
    ]
    
    # Phishing patterns
    phishing_patterns = [
        'secure-{domain}', '{domain}-update', '{domain}-verify', 'login-{domain}',
        '{domain}-security', 'account-{domain}', '{domain}-signin', 'verify-{domain}'
    ]
    
    urls = []
    labels = []
    
    # Generate legitimate URLs (50%)
    for _ in range(n_samples // 2):
        domain = np.random.choice(legitimate_domains)
        protocol = np.random.choice(['http', 'https'], p=[0.2, 0.8])
        paths = ['', '/home', '/about', '/contact', '/products']
        path = np.random.choice(paths)
        url = f"{protocol}://{domain}{path}"
        urls.append(url)
        labels.append(0)  # Legitimate
    
    # Generate phishing URLs (50%)
    for _ in range(n_samples // 2):
        base_domain = np.random.choice(legitimate_domains)
        pattern = np.random.choice(phishing_patterns)
        suspicious_domain = pattern.format(domain=base_domain.replace('.', '-'))
        
        # Add suspicious TLDs
        tld = np.random.choice(['.tk', '.ml', '.ga', '.cf', '.com'])
        suspicious_domain += tld
        
        protocol = np.random.choice(['http', 'https'], p=[0.7, 0.3])
        paths = ['/login', '/secure', '/update', '/verify', '/account']
        path = np.random.choice(paths)
        
        url = f"{protocol}://{suspicious_domain}{path}"
        urls.append(url)
        labels.append(1)  # Phishing
    
    # Create DataFrame
    df = pd.DataFrame({'url': urls, 'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    # Save dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/phishing_dataset.csv', index=False)
    
    print(f"✅ Created dataset with {len(df)} samples")
    print(f"   - Legitimate: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    print(f"   - Phishing: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    create_sample_dataset()