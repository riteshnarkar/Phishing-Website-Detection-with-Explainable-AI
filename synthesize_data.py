import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/sample_dataset_features.csv')

print(f"Original missing domain_age count: {(df['domain_age'] == -1).sum()} out of {len(df)}")

np.random.seed(42)

# Synthesize rules:
# Legitimate sites (label 0): generally have older domains (e.g. 1 year to 15 years), valid SSL, MX records
# Phishing sites (label 1): generally have much newer domains (e.g. 1 day to 45 days), missing MX, missing/invalid SSL

# Generate values for Legitimate Sites (label == 0)
legit_mask = df['label'] == 0
n_legit = legit_mask.sum()
df.loc[legit_mask, 'domain_age'] = np.random.randint(365, 5000, size=n_legit)
legit_ssl_valid = np.random.choice([0, 1], size=n_legit, p=[0.05, 0.95])
df.loc[legit_mask, 'has_ssl'] = legit_ssl_valid
df.loc[legit_mask, 'ssl_valid'] = legit_ssl_valid
# Most legit have MX, some have SPF/DMARC
df.loc[legit_mask, 'has_mx_record'] = np.random.choice([0, 1], size=n_legit, p=[0.1, 0.9])

# Generate values for Phishing Sites (label == 1)
phish_mask = df['label'] == 1
n_phish = phish_mask.sum()
df.loc[phish_mask, 'domain_age'] = np.random.randint(1, 60, size=n_phish)
# Less likely to have valid SSL
phish_ssl_valid = np.random.choice([0, 1], size=n_phish, p=[0.6, 0.4])
df.loc[phish_mask, 'has_ssl'] = np.random.choice([0, 1], size=n_phish, p=[0.2, 0.8])
df.loc[phish_mask, 'ssl_valid'] = df.loc[phish_mask, 'has_ssl'] & phish_ssl_valid
# Less likely to have MX
df.loc[phish_mask, 'has_mx_record'] = np.random.choice([0, 1], size=n_phish, p=[0.7, 0.3])

# Save synthetic dataset
df.to_csv('data/dataset_features_synthetic.csv', index=False)
print("Saved synthesized dataset to data/dataset_features_synthetic.csv")
