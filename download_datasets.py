import requests
import pandas as pd
import os
from urllib.parse import urlparse

def download_uci_dataset():
    """Download UCI Phishing Website Dataset."""
    print("📥 Downloading UCI Phishing Dataset...")
    
    # UCI dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save raw file
        os.makedirs('data/raw', exist_ok=True)
        with open('data/raw/uci_phishing.arff', 'wb') as f:
            f.write(response.content)
        
        print("✅ UCI dataset downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download UCI dataset: {e}")
        return False

def download_phishtank_sample():
    """Download sample from PhishTank (requires internet)."""
    print("📥 Downloading PhishTank sample...")
    
    # Note: Full PhishTank requires API key
    # This is a sample approach - you may need to modify
    sample_phishing_urls = [
        "http://paypal-security.tk/login",
        "http://amazon-update.ml/verify", 
        "http://microsoft-account.ga/signin",
        "http://facebook-security.cf/confirm",
        "http://instagram-verify.tk/account"
    ]
    
    sample_legitimate_urls = [
        "https://www.paypal.com",
        "https://www.amazon.com",
        "https://www.microsoft.com",
        "https://www.facebook.com", 
        "https://www.instagram.com"
    ]
    
    urls = sample_phishing_urls + sample_legitimate_urls
    labels = [1] * len(sample_phishing_urls) + [0] * len(sample_legitimate_urls)
    
    df = pd.DataFrame({'url': urls, 'label': labels})
    df.to_csv('data/raw/phishtank_sample.csv', index=False)
    
    print("✅ PhishTank sample created")
    return True

if __name__ == "__main__":
    download_uci_dataset()
    download_phishtank_sample()