"""
Feature Extractor for Phishing Website Detection
Extracts URL-based, content-based, and host-based features for ML models.
"""

import re
import socket
import ssl
import requests
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional DNS/WHOIS imports - these may not be installed
try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False
    print("WARNING: dnspython not installed. DNS features will use defaults.")

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    print("WARNING: whois/python-whois not installed. Domain age features will use defaults.")

class FeatureExtractor:
    """
    Comprehensive feature extractor for phishing detection.
    Extracts URL, content, and host-based features.
    """
    
    def __init__(self, timeout=10):
        """
        Initialize feature extractor.
        
        Args:
            timeout (int): Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_all_features(self, url):
        """
        Extract all features from a given URL.
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        features = {}
        
        # URL-based features (always extractable)
        features.update(self._extract_url_features(url))
        
        # Content-based features (require HTTP request)
        try:
            html_content = self._fetch_content(url)
            features.update(self._extract_content_features(html_content, url))
        except Exception as e:
            print(f"Warning: Could not fetch content for {url}: {str(e)}")
            features.update(self._get_default_content_features())
        
        # Host-based features (require domain analysis)
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            features.update(self._extract_host_features(domain, url))
        except Exception as e:
            print(f"Warning: Could not extract host features for {url}: {str(e)}")
            features.update(self._get_default_host_features())
        
        return features
    
    def _extract_url_features(self, url):
        """Extract URL-based features."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path
        query = parsed_url.query
        
        features = {
            # Basic length features
            'url_length': len(url),
            'domain_length': len(domain),
            'path_length': len(path),
            
            # Character analysis
            'special_char_count': len(re.findall(r'[^a-zA-Z0-9.-]', url)),
            'dots_count': url.count('.'),
            'hyphens_count': url.count('-'),
            'underscores_count': url.count('_'),
            'digits_count': len(re.findall(r'\d', url)),
            
            # Protocol and structure
            'has_https': 1 if parsed_url.scheme == 'https' else 0,
            'has_www': 1 if domain.startswith('www.') else 0,
            'has_at_symbol': 1 if '@' in url else 0,
            'has_ip_address': 1 if self._is_ip_address(domain) else 0,
            
            # URL structure analysis
            'subdomain_count': len(domain.split('.')) - 2 if '.' in domain else 0,
            'url_depth': len([x for x in path.split('/') if x]),
            'has_port': 1 if ':' in domain and not domain.startswith('www.') else 0,
            
            # Query parameters
            'query_length': len(query),
            'params_count': len(parse_qs(query)),
            
            # Suspicious patterns
            'has_suspicious_words': self._has_suspicious_words(url),
            'entropy': self._calculate_entropy(url)
        }
        
        return features
    
    def _extract_content_features(self, html_content, url):
        """Extract content-based features."""
        if not html_content:
            return self._get_default_content_features()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Forms analysis
        forms = soup.find_all('form')
        input_fields = soup.find_all('input')
        
        # Links analysis
        all_links = soup.find_all('a', href=True)
        external_links = self._count_external_links(all_links, url)
        
        # Media analysis
        images = soup.find_all('img')
        scripts = soup.find_all('script')
        iframes = soup.find_all('iframe')
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        title_tag = soup.find('title')
        favicon = soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')
        
        features = {
            # Forms and inputs
            'has_login_form': self._has_login_form(forms),
            'forms_count': len(forms),
            'input_fields_count': len(input_fields),
            
            # Content structure
            'has_javascript': 1 if scripts else 0,
            'scripts_count': len(scripts),
            'has_iframes': 1 if iframes else 0,
            'iframes_count': len(iframes),
            
            # Links analysis
            'total_links': len(all_links),
            'external_links': external_links,
            'external_links_ratio': external_links / max(len(all_links), 1),
            
            # Media content
            'images_count': len(images),
            'has_favicon': 1 if favicon else 0,
            
            # Page characteristics
            'content_length': len(html_content),
            'has_title': 1 if title_tag and title_tag.string else 0,
            'meta_tags_count': len(meta_tags),
            
            # Suspicious patterns
            'has_meta_refresh': self._has_meta_refresh(meta_tags),
            'has_popup_patterns': self._has_popup_patterns(html_content),
            'has_suspicious_js': self._has_suspicious_javascript(scripts)
        }
        
        return features
    
    def _extract_host_features(self, domain, url):
        """Extract host-based features."""
        features = {}
        
        # Strip port from domain if present (e.g. "example.com:8080" -> "example.com")
        clean_domain = domain.split(':')[0] if ':' in domain else domain
        # Strip www. prefix for WHOIS lookups
        whois_domain = clean_domain
        if whois_domain.startswith('www.'):
            whois_domain = whois_domain[4:]
        
        # Domain registration info
        if WHOIS_AVAILABLE:
            try:
                # Try python-whois style first (whois.whois())
                try:
                    domain_info = whois.whois(whois_domain)
                except AttributeError:
                    # Fallback for WhoisDomain-style package (whois.query())
                    try:
                        domain_info = whois.query(whois_domain)
                    except Exception:
                        domain_info = None
                
                if domain_info:
                    creation_date = getattr(domain_info, 'creation_date', None)
                    expiration_date = getattr(domain_info, 'expiration_date', None)
                    
                    if isinstance(creation_date, list):
                        creation_date = creation_date[0]
                    if isinstance(expiration_date, list):
                        expiration_date = expiration_date[0]
                    
                    # Make dates timezone-naive for comparison with datetime.now()
                    if creation_date and hasattr(creation_date, 'tzinfo') and creation_date.tzinfo:
                        creation_date = creation_date.replace(tzinfo=None)
                    if expiration_date and hasattr(expiration_date, 'tzinfo') and expiration_date.tzinfo:
                        expiration_date = expiration_date.replace(tzinfo=None)
                    
                    if creation_date:
                        domain_age = (datetime.now() - creation_date).days
                        features['domain_age'] = max(domain_age, 0)
                    else:
                        features['domain_age'] = -1
                        
                    if expiration_date:
                        days_to_expire = (expiration_date - datetime.now()).days
                        features['domain_expiry_days'] = max(days_to_expire, 0)
                    else:
                        features['domain_expiry_days'] = -1
                else:
                    features['domain_age'] = -1
                    features['domain_expiry_days'] = -1
                    
            except Exception as e:
                print(f"Warning: WHOIS lookup failed for {whois_domain}: {str(e)[:100]}")
                features['domain_age'] = -1
                features['domain_expiry_days'] = -1
        else:
            features['domain_age'] = -1
            features['domain_expiry_days'] = -1
        
        # SSL Certificate analysis
        ssl_info = self._get_ssl_info(clean_domain, url)
        features.update(ssl_info)
        
        # DNS records analysis
        dns_info = self._get_dns_info(clean_domain)
        features.update(dns_info)
        
        return features
    
    def _get_ssl_info(self, domain, url):
        """Get SSL certificate information."""
        features = {
            'has_ssl': 0,
            'ssl_age': -1,
            'ssl_valid': 0
        }
        
        if not url.startswith('https://'):
            return features
        
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    
                    features['has_ssl'] = 1
                    
                    # Check certificate validity
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                    
                    now = datetime.now()
                    if not_before <= now <= not_after:
                        features['ssl_valid'] = 1
                    
                    # SSL certificate age
                    ssl_age = (now - not_before).days
                    features['ssl_age'] = max(ssl_age, 0)
                    
        except Exception:
            pass
          
        return features
    
    def _get_dns_info(self, domain):
        """Get DNS information."""
        features = {
            'dns_records_count': 0,
            'has_mx_record': 0,
            'has_spf_record': 0,
            'has_dmarc_record': 0
        }
        
        if not DNS_AVAILABLE:
            return features
        
        # Use a shorter timeout for DNS queries to avoid hanging
        resolver = dns.resolver.Resolver()
        resolver.timeout = 3
        resolver.lifetime = 3
        
        try:
            # Count different DNS record types
            record_types = ['A', 'AAAA', 'CNAME', 'NS']
            total_records = 0
            
            for record_type in record_types:
                try:
                    answers = resolver.resolve(domain, record_type)
                    total_records += len(answers)
                except Exception:
                    continue
            
            features['dns_records_count'] = total_records
            
            # Check for MX record
            try:
                resolver.resolve(domain, 'MX')
                features['has_mx_record'] = 1
            except Exception:
                pass
            
            # Check for SPF record
            try:
                txt_records = resolver.resolve(domain, 'TXT')
                for record in txt_records:
                    if 'v=spf1' in str(record).lower():
                        features['has_spf_record'] = 1
                        break
            except Exception:
                pass
            
            # Check for DMARC record
            try:
                dmarc_domain = f'_dmarc.{domain}'
                txt_records = resolver.resolve(dmarc_domain, 'TXT')
                for record in txt_records:
                    if 'v=dmarc1' in str(record).lower():
                        features['has_dmarc_record'] = 1
                        break
            except Exception:
                pass
                
        except Exception as e:
            print(f"Warning: DNS lookup failed for {domain}: {str(e)[:100]}")
        
        return features
    
    def _fetch_content(self, url):
        """Fetch HTML content from URL."""
        try:
            response = self.session.get(url, timeout=self.timeout, verify=False)
            response.raise_for_status()
            return response.text
        except Exception:
            return None
    
    def _is_ip_address(self, domain):
        """Check if domain is an IP address."""
        ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        return bool(re.match(ip_pattern, domain))
    
    def _has_suspicious_words(self, url):
        """Check for suspicious keywords in URL."""
        suspicious_words = [
            'secure', 'account', 'update', 'login', 'signin', 'banking',
            'paypal', 'amazon', 'microsoft', 'apple', 'google', 'verify',
            'suspended', 'confirm', 'urgent', 'security', 'alert'
        ]
        
        url_lower = url.lower()
        count = sum(1 for word in suspicious_words if word in url_lower)
        return count
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text."""
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        entropy = -sum([p * np.log2(p) for p in prob])
        return entropy
    
    def _has_login_form(self, forms):
        """Check if any form looks like a login form."""
        for form in forms:
            inputs = form.find_all('input')
            input_types = [inp.get('type', '').lower() for inp in inputs]
            
            # Look for password field
            if 'password' in input_types:
                return 1
            
            # Look for common login field names
            input_names = [inp.get('name', '').lower() for inp in inputs]
            login_indicators = ['username', 'email', 'login', 'user', 'password', 'pass']
            
            if any(indicator in ' '.join(input_names) for indicator in login_indicators):
                return 1
        
        return 0
    
    def _count_external_links(self, links, base_url):
        """Count external links."""
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc.lower()
        
        external_count = 0
        for link in links:
            href = link.get('href', '')
            if href.startswith('http'):
                parsed_link = urlparse(href)
                if parsed_link.netloc.lower() != base_domain:
                    external_count += 1
        
        return external_count
    
    def _has_meta_refresh(self, meta_tags):
        """Check for meta refresh redirect."""
        for meta in meta_tags:
            if meta.get('http-equiv', '').lower() == 'refresh':
                return 1
        return 0
    
    def _has_popup_patterns(self, html_content):
        """Check for popup patterns in HTML."""
        popup_patterns = [
            'window.open', 'popup', 'alert(', 'confirm(',
            'onload=', 'onclick=', 'javascript:'
        ]
        
        html_lower = html_content.lower()
        return sum(1 for pattern in popup_patterns if pattern in html_lower)
    
    def _has_suspicious_javascript(self, scripts):
        """Check for suspicious JavaScript patterns."""
        suspicious_patterns = [
            'eval(', 'unescape(', 'document.write(',
            'window.location', 'fromCharCode', 'atob(',
            'btoa(', 'escape('
        ]
        
        suspicious_count = 0
        for script in scripts:
            if script.string:
                script_content = script.string.lower()
                suspicious_count += sum(1 for pattern in suspicious_patterns 
                                     if pattern in script_content)
        
        return suspicious_count
    
    def _get_default_content_features(self):
        """Return default content features when content cannot be fetched."""
        return {
            'has_login_form': 0, 'forms_count': 0, 'input_fields_count': 0,
            'has_javascript': 0, 'scripts_count': 0, 'has_iframes': 0,
            'iframes_count': 0, 'total_links': 0, 'external_links': 0,
            'external_links_ratio': 0, 'images_count': 0, 'has_favicon': 0,
            'content_length': 0, 'has_title': 0, 'meta_tags_count': 0,
            'has_meta_refresh': 0, 'has_popup_patterns': 0, 'has_suspicious_js': 0
        }
    
    def _get_default_host_features(self):
        """Return default host features when analysis fails."""
        return {
            'domain_age': -1, 'domain_expiry_days': -1, 'has_ssl': 0,
            'ssl_age': -1, 'ssl_valid': 0, 'dns_records_count': 0,
            'has_mx_record': 0, 'has_spf_record': 0, 'has_dmarc_record': 0
        }

def extract_features_batch(urls, output_file=None):
    """
    Extract features for a batch of URLs.
    
    Args:
        urls (list): List of URLs to process
        output_file (str): Optional CSV file to save results
        
    Returns:
        pandas.DataFrame: DataFrame with extracted features
    """
    extractor = FeatureExtractor()
    features_list = []
    
    print(f"Processing {len(urls)} URLs...")
    
    for i, url in enumerate(urls):
        try:
            print(f"Processing {i+1}/{len(urls)}: {url}")
            features = extractor.extract_all_features(url)
            features['url'] = url
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    df = pd.DataFrame(features_list)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Example usage
    test_urls = [
        "https://www.google.com",
        "http://phishing-example.com/secure/login",
        "https://github.com/user/repo"
    ]
    
    extractor = FeatureExtractor()
    
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        features = extractor.extract_all_features(url)
        
        print("Key features:")
        for key, value in list(features.items())[:10]:
            print(f"  {key}: {value}")
        
        print(f"Total features extracted: {len(features)}")