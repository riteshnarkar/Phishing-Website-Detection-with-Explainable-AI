"""
Enhanced Explainer Module for Phishing Detection
Provides detailed, multi-section explanations for ML model predictions.
"""

import numpy as np
import pandas as pd
from urllib.parse import urlparse
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PhishingExplainer:
    """
    Enhanced explainer class that generates detailed explanations 
    for phishing detection predictions.
    """
    
    def __init__(self, model, scaler, feature_names):
        """
        Initialize the explainer.
        
        Args:
            model: Trained ML model
            scaler: Feature scaler (can be None)
            feature_names: List of feature names
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        
        # Define suspicious patterns and keywords
        self.suspicious_keywords = [
            'login', 'signin', 'account', 'verify', 'suspended', 'update',
            'confirm', 'secure', 'urgent', 'immediately', 'click', 'winner',
            'free', 'prize', 'congratulations', 'limited', 'offer', 'deal',
            'paypal', 'amazon', 'microsoft', 'google', 'apple', 'bank',
            'alert', 'warning', 'expire', 'activate', 'validate', 'exchange'
        ]
        
        self.legitimate_tlds = [
            'com', 'org', 'net', 'edu', 'gov', 'mil', 'int'
        ]
        
        self.suspicious_tlds = [
            'tk', 'ml', 'ga', 'cf', 'top', 'click', 'download', 'bid',
            'country', 'stream', 'accountant', 'science', 'work'
        ]
        
        self.url_shorteners = [
            'bit.ly', 'tinyurl', 't.co', 'goo.gl', 'short.link', 'ow.ly',
            'buff.ly', 'adf.ly', 'is.gd', 'v.gd', 'tiny.cc'
        ]
        
        self.brand_keywords = [
            'paypal', 'amazon', 'microsoft', 'google', 'apple', 'facebook',
            'instagram', 'twitter', 'linkedin', 'ebay', 'alibaba', 'bank'
        ]
    
    def create_explanation_report(self, features_array, url, explanation_level='detailed'):
        """
        Create comprehensive explanation report for a prediction.
        
        Args:
            features_array: Feature array used for prediction
            url: Original URL
            explanation_level: Level of detail ('basic', 'detailed', 'comprehensive')
            
        Returns:
            dict: Explanation report with multiple sections
        """
        # Make prediction and get probabilities
        if self.scaler:
            features_scaled = self.scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        prediction = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Calculate confidence
        phishing_prob = prediction_proba[1]
        confidence = phishing_prob if prediction == 1 else (1 - phishing_prob)
        
        prediction_label = 'phishing' if prediction == 1 else 'legitimate'
        
        # Create features dictionary
        features_dict = {
            name: float(features_array[0][i]) 
            for i, name in enumerate(self.feature_names)
        }
        
        # Always use the standardized feature-based explanation
        return self._create_feature_based_explanation(url, prediction_label, confidence, features_dict)
    
    def _create_feature_based_explanation(self, url, prediction, confidence, features_dict):
        """
        Create a standardized explanation highlighting top 4-5 key reasons.
        Replaces all previous explanation levels.
        """
        # Generate simplified summary
        summary = self._generate_simplified_summary(prediction, confidence, url)
        
        # Get key reasons (4-5 bullet points)
        key_reasons = self._identify_key_reasons(url, prediction, features_dict)
        
        # Structure as sections for frontend compatibility
        sections = [
            {
                'title': 'Key Analysis Factors',
                'content': key_reasons
            },
            {
                'title': 'Security Recommendation',
                'content': self._get_simplified_advice(prediction)
            }
        ]
        
        detailed_explanation = {
            'summary': summary,
            'sections': sections
        }
        
        return {
            'explanation_text': summary,
            'detailed_explanation': detailed_explanation,
            'confidence': confidence,
            'prediction': prediction,
            'feature_importance': self._get_feature_importance(features_dict)
        }

    def _identify_key_reasons(self, url, prediction, features_dict):
        """
        Identify specific reasons based on ML features.
        Maps internal feature names to user-friendly explanations.
        """
        reasons = []
        parsed = urlparse(url)
        
        # --- 1. CRITICAL FEATURES (High Risk) ---
        
        # IP Address
        if features_dict.get('has_ip_address', 0) == 1:
            reasons.append("⚠️ <b>IP Address URL:</b> Using an IP address instead of a domain is a top phishing indicator.")

        # Login Forms
        if features_dict.get('has_login_form', 0) == 1:
            reasons.append("⚠️ <b>Login Request:</b> The site asks for credentials (username/password), typical of phishing.")

        # Popups
        if features_dict.get('has_popup_patterns', 0) > 0:
             reasons.append("⚠️ <b>Aggressive Popups:</b> Code for intrusive popups detected, often used to trap users.")
             
        # Suspicious JavaScript
        if features_dict.get('has_suspicious_js', 0) > 0:
             reasons.append("⚠️ <b>Malicious Code:</b> Suspicious JavaScript patterns detected (e.g., eval, unescape).")
             
        # Meta Refresh (Auto-redirect)
        if features_dict.get('has_meta_refresh', 0) == 1:
             reasons.append("⚠️ <b>Sneaky Redirect:</b> Page attempts to automatically redirect you to another site (Meta Refresh).")


        # --- 2. HOST & DOMAIN FEATURES ---
        
        # Domain Age (Freshly registered domains are risky)
        age = features_dict.get('domain_age', -1)
        if age != -1 and age < 30:
             reasons.append(f"⚠️ <b>New Domain:</b> Registered only {age} days ago. Phishing sites are often brand new.")

        # SSL/TLS
        if features_dict.get('has_ssl', 0) == 0:
            reasons.append("⚠️ <b>No Encryption:</b> Site does not use HTTPS/SSL. Data is sent in plain text.")
        elif features_dict.get('ssl_valid', 0) == 0 and features_dict.get('has_ssl', 0) == 1:
             reasons.append("⚠️ <b>Invalid SSL:</b> The security certificate is invalid or expired.")

        # DNS Records
        if features_dict.get('has_mx_record', 0) == 0 and features_dict.get('has_ssl', 0) == 1:
             reasons.append("⚠️ <b>Missing Email Records:</b> Domain cannot receive emails (No MX record), suggesting it's disposable.")

        # Suspicious TLD Check (Moved to Domain Features for higher priority)
        risk_tlds = [
            '.xyz', '.top', '.club', '.info', '.site', '.cn', '.tk', '.ga', '.ml', '.work', 
            '.click', '.link', '.online', '.store', '.shop', '.live', '.buzz', '.vip', 
            '.win', '.bid', '.pro', '.cc', '.fun', '.rest', '.bar', '.cam'
        ]
        for tld in risk_tlds:
            if parsed.netloc.lower().endswith(tld):
                reasons.append(f"⚠️ <b>Suspicious TLD ({tld}):</b> '{tld}' is a high-risk domain extension often abused by scammers.")
                break # Only show one TLD warning


        # --- 3. URL STRUCTURE & CONTENT ---
        
        # URL Length (High Risk if very long)
        url_len = len(url)
        if url_len > 50:
             reasons.append(f"⚠️ <b>Long URL ({url_len} chars):</b> Excessively long URLs are often used to hide suspicious patterns.")

        # Suspicious Words
        susp_word_count = features_dict.get('has_suspicious_words', 0)
        if susp_word_count > 0:
             # Identify actual words for the explanation
             found_words = []
             check_list = [
                'secure', 'account', 'update', 'login', 'signin', 'banking',
                'paypal', 'amazon', 'microsoft', 'apple', 'google', 'verify',
                'suspended', 'confirm', 'urgent', 'security', 'alert'
             ]
             for word in check_list:
                 if word in url.lower():
                     found_words.append(word)
                     if len(found_words) >= 3: break # Limit to 3
             
             display_words = ", ".join([f"'{w}'" for w in found_words])
             reasons.append(f"⚠️ <b>Suspicious Keywords:</b> URL contains alarm word(s): {display_words}.")
        
        # Subdomains
        if features_dict.get('subdomain_count', 0) > 2:
             reasons.append("⚠️ <b>Complex Subdomains:</b> Multiple subdomains used, possibly to mimic a deeper legitimate path.")
        elif features_dict.get('dots_count', 0) > 3:
             reasons.append(f"⚠️ <b>Complex Domain:</b> Contains {int(features_dict.get('dots_count'))} dots, which is unusually high and confusing.")
             
        # @ Symbol Obfuscation
        if features_dict.get('has_at_symbol', 0) == 1:
             reasons.append("⚠️ <b>URL Obfuscation:</b> Contains '@' symbol, often used to trick browsers into ignoring the first part of the URL.")

        # High Entropy (Randomness)
        if features_dict.get('entropy', 0) > 4.5: # Threshold for high randomness
             reasons.append("⚠️ <b>Randomized URL:</b> The URL contains random characters, typical of generated phishing links.")

        # Non-Standard Port
        if features_dict.get('has_port', 0) == 1:
             reasons.append("⚠️ <b>Non-Standard Port:</b> Connects to a custom port number, often used to bypass firewalls.")

        # Tiny URL
        if features_dict.get('url_length', 0) < 25 and features_dict.get('has_https', 0) == 0:
             reasons.append("⚠️ <b>Suspicious Shortener:</b> Unusually short URL without SSL, possibly a redirector.")

        # Content - External Links
        ext_ratio = features_dict.get('external_links_ratio', 0)
        if ext_ratio > 0.8:
             reasons.append("⚠️ <b>External Links:</b> Most links point to other domains, a tactic to make a fake page look functional.")
             
        # Missing Favicon (often missing on quick phishing sites)
        if features_dict.get('has_favicon', 0) == 0 and prediction == 'phishing':
             reasons.append("⚠️ <b>Missing Identity:</b> No website icon (favicon) detected, common in low-effort phishing pages.")


        # --- 4. BRAND & PATTERN FALLBACKS ---
        
        parsed = urlparse(url)
        
        # Brand Check
        common_brands = [
            'paypal', 'google', 'facebook', 'netflix', 'apple', 'microsoft', 'amazon', 
            'chase', 'wellsfargo', 'bofa', 'citibank', 'dropbox', 'adobe', 'linkedin'
        ]
        for brand in common_brands:
            if brand in url.lower() and brand not in parsed.netloc:
                 if f"Brand Impersonation" not in str(reasons):
                     reasons.append(f"⚠️ <b>Brand Impersonation:</b> '{brand}' is in the URL path but not the domain.")

        # (TLD Check moved to Section 2)

        
        # --- 5. LEGITIMATE SIGNALS (If Safe) ---
        if prediction == 'legitimate':
             clean_reasons = []
             if features_dict.get('has_ssl', 0) == 1:
                 clean_reasons.append("✅ <b>Valid SSL:</b> Connection is encrypted and certificate is valid.")
             if features_dict.get('domain_age', 0) > 365:
                 age_yrs = int(features_dict.get('domain_age')/365)
                 clean_reasons.append(f"✅ <b>Established Domain:</b> Registered over {age_yrs} years ago.")
             if features_dict.get('has_mx_record', 0) == 1:
                 clean_reasons.append("✅ <b>Email Verified:</b> Domain is set up to receive emails (MX record found).")
             if features_dict.get('has_suspicious_words', 0) == 0:
                 clean_reasons.append("✅ <b>Clean URL:</b> No suspicious keywords found.")
             
             reasons = clean_reasons[:4]

        # --- 6. LAST RESORT (Only if EMPTY) ---
        if not reasons and prediction == 'phishing':
             reasons.append("⚠️ <b>AI Analysis:</b> The model detected a combination of minor risk factors (e.g., structure, content) that resemble known phishing patterns, even though no single critical flaw was found.")

        return reasons[:5]
    
    def _create_simplified_explanation(self, url, prediction, confidence, features_dict):
        """Create simplified, jargon-free explanation."""
        
        # Generate simple summary
        summary = self._generate_simplified_summary(prediction, confidence, url)
        
        # Create sections
        sections = []
        
        # Section 1: The "Why"
        sections.append({
            'title': 'Why is this result?',
            'content': self._get_simplified_reasons(url, prediction, confidence, features_dict)
        })
        
        # Section 2: Actionable Advice
        sections.append({
            'title': 'What should you do?',
            'content': self._get_simplified_advice(prediction)
        })
        
        detailed_explanation = {
            'summary': summary,
            'sections': sections
        }
        
        return {
            'explanation_text': summary,
            'detailed_explanation': detailed_explanation,
            'confidence': confidence,
            'prediction': prediction,
            'feature_importance': self._get_feature_importance(features_dict)
        }

    def _generate_simplified_summary(self, prediction, confidence, url):
        """Generate a very simple summary sentence."""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if prediction == 'phishing':
            if confidence > 0.8:
                return f"⚠️ DANGER: This website ({domain}) looks definitely unsafe."
            else:
                return f"⚠️ CAUTION: This website ({domain}) looks suspicious."
        else:
            if confidence > 0.8:
                return f"✅ SAFE: This website ({domain}) looks real and safe."
            else:
                return f"✅ PROBABLY SAFE: This website ({domain}) seems okay, but be careful."

    def _get_simplified_reasons(self, url, prediction, confidence, features_dict):
        """Get bullet points suitable for non-technical users."""
        reasons = []
        
        parsed = urlparse(url)
        
        # HTTPS check (easy to explain)
        if not url.startswith('https'):
            reasons.append("It does not use a secure connection (no lock icon).")
        else:
            if prediction == 'legitimate':
                reasons.append("It uses a secure connection (has the lock icon).")
        
        # Length check
        if len(url) > 75:
            reasons.append("The web address is unusually long, which is often a trick.")
            
        # Dots check (subdomains)
        if features_dict.get('num_dots', 0) > 3:
            reasons.append("The web address is complicated and confusing.")
            
        # Suspicious keywords
        for kw in self.suspicious_keywords:
            if kw in url.lower():
                reasons.append(f"It contains the word '{kw}', which scammers often use.")
                break # Just mention one to keep it simple
                
        # IP address check
        if re.match(r'^\d+\.\d+\.\d+\.\d+', parsed.netloc):
             reasons.append("It uses numbers instead of a name, which is very suspicious.")

        # Default fallback if list is empty
        if not reasons:
            if prediction == 'phishing':
                reasons.append("Our AI noticed hidden patterns often used by scammers.")
            else:
                reasons.append("The web address follows standard safe patterns.")
                
        return reasons

    def _get_simplified_advice(self, prediction):
        """Get clear, single-sentence advice."""
        if prediction == 'phishing':
            return [
                "Do not enter your password or credit card info.",
                "Close this page immediately.",
                "If you are unsure, search for the official website on Google."
            ]
        else:
            return [
                "You can likely browse this site safely.",
                "Still, never share your password unless you are sure.",
                "Make sure the website address looks correct."
            ]
    
    def _create_basic_explanation(self, url, prediction, confidence, features_dict):
        """Create basic explanation (original format)."""
        confidence_pct = confidence * 100
        
        if prediction == 'phishing':
            explanation_text = f"This URL is classified as PHISHING with {confidence_pct:.1f}% confidence. The analysis detected suspicious patterns commonly associated with malicious websites. Avoid this URL and do not enter personal information."
        else:
            explanation_text = f"This URL appears LEGITIMATE with {confidence_pct:.1f}% confidence. The analysis found indicators of a trustworthy website with minimal risk factors. Exercise normal web browsing caution."
        
        return {
            'explanation_text': explanation_text,
            'confidence': confidence,
            'prediction': prediction,
            'feature_importance': self._get_feature_importance(features_dict)
        }
    
    def _create_detailed_explanation(self, url, prediction, confidence, features_dict, explanation_level):
        """Create detailed multi-section explanation."""
        
        # Generate comprehensive summary
        summary = self._generate_detailed_summary(prediction, confidence, url)
        
        # Create sections based on explanation level
        sections = []
        
        if explanation_level == 'detailed':
            # DETAILED: Core 4 sections for balanced analysis
            sections.append({
                'title': 'URL Structure Analysis',
                'content': self._analyze_url_structure(url, features_dict)
            })
            
            sections.append({
                'title': 'Risk Factor Assessment',
                'content': self._identify_risk_factors(url, prediction, confidence, features_dict)
            })
            
            sections.append({
                'title': 'Security Recommendations',
                'content': self._get_security_recommendations(prediction, confidence)
            })
            
            sections.append({
                'title': 'Technical Analysis',
                'content': self._get_technical_breakdown(features_dict, prediction, confidence)
            })
            
        elif explanation_level == 'comprehensive':
            # COMPREHENSIVE: All sections + advanced analysis + domain intelligence
            sections.append({
                'title': 'Advanced URL Intelligence',
                'content': self._analyze_advanced_url_intelligence(url, features_dict)
            })
            
            sections.append({
                'title': 'Comprehensive Risk Assessment',
                'content': self._comprehensive_risk_assessment(url, prediction, confidence, features_dict)
            })
            
            sections.append({
                'title': 'Security Posture Analysis',
                'content': self._analyze_security_posture(url, features_dict, prediction)
            })
            
            sections.append({
                'title': 'Machine Learning Deep Analysis',
                'content': self._ml_deep_analysis(features_dict, prediction, confidence)
            })
            
            sections.append({
                'title': 'Threat Intelligence & Context',
                'content': self._threat_intelligence_analysis(url, prediction, confidence)
            })
            
            sections.append({
                'title': 'Comprehensive Security Recommendations',
                'content': self._comprehensive_security_recommendations(prediction, confidence, url)
            })
            
            sections.append({
                'title': 'Technical Deep Dive',
                'content': self._technical_deep_dive(features_dict, url, prediction)
            })
        
        # Create the detailed explanation structure
        detailed_explanation = {
            'summary': summary,
            'sections': sections
        }
        
        return {
            'explanation_text': summary,  # For backward compatibility
            'detailed_explanation': detailed_explanation,
            'confidence': confidence,
            'prediction': prediction,
            'feature_importance': self._get_feature_importance(features_dict)
        }
    
    def _generate_detailed_summary(self, prediction, confidence, url):
        """Generate comprehensive summary explanation."""
        confidence_pct = confidence * 100
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if prediction == 'phishing':
            if confidence > 0.95:
                return f"CRITICAL THREAT DETECTED: This URL ({domain}) is classified as PHISHING with extremely high confidence ({confidence_pct:.1f}%). The machine learning analysis identified multiple strong indicators of malicious intent, including suspicious URL patterns, potentially deceptive domain structures, and characteristics commonly associated with phishing attacks. This represents a severe security risk and should be avoided completely."
            elif confidence > 0.85:
                return f"HIGH RISK PHISHING: This URL ({domain}) is classified as PHISHING with very high confidence ({confidence_pct:.1f}%). The comprehensive analysis revealed several concerning patterns that strongly suggest this is a malicious website designed to steal personal information, credentials, or conduct fraudulent activities. Exercise extreme caution and avoid this URL."
            elif confidence > 0.7:
                return f"PHISHING DETECTED: This URL ({domain}) is classified as PHISHING with high confidence ({confidence_pct:.1f}%). Multiple suspicious indicators were detected that commonly appear in phishing attempts. While not as definitive as higher confidence predictions, the risk is still significant enough to warrant complete avoidance."
            else:
                return f"POTENTIAL PHISHING: This URL ({domain}) is classified as PHISHING with moderate confidence ({confidence_pct:.1f}%). Some concerning patterns were detected, but the signals are mixed. The model suggests caution, though there's more uncertainty in this classification than in higher-confidence predictions."
        else:
            if confidence > 0.95:
                return f"HIGHLY TRUSTED: This URL ({domain}) appears LEGITIMATE with extremely high confidence ({confidence_pct:.1f}%). The comprehensive analysis found strong indicators of a trustworthy website with minimal risk factors. The URL structure, domain characteristics, and technical features all align with legitimate web properties."
            elif confidence > 0.85:
                return f"TRUSTED: This URL ({domain}) appears LEGITIMATE with very high confidence ({confidence_pct:.1f}%). Most indicators strongly suggest this is a safe website from a reputable source. The analysis found minimal red flags and multiple positive trust signals."
            elif confidence > 0.7:
                return f"LIKELY SAFE: This URL ({domain}) appears LEGITIMATE with high confidence ({confidence_pct:.1f}%). The majority of indicators suggest this is a safe website, though some minor concerns may exist. Overall assessment indicates low risk for typical browsing activities."
            else:
                return f"UNCERTAIN BUT LIKELY SAFE: This URL ({domain}) appears LEGITIMATE with moderate confidence ({confidence_pct:.1f}%). While classified as safe, there are some ambiguous signals that create uncertainty. The site is likely legitimate but exercise normal web browsing caution."
    
    def _analyze_url_structure(self, url, features_dict):
        """Perform detailed URL structure analysis."""
        parsed = urlparse(url)
        analysis = []
        
        # Protocol analysis
        analysis.append(f"Protocol: {parsed.scheme.upper()} {'(Secure Encrypted Connection)' if parsed.scheme == 'https' else '(Unencrypted - Security Risk)'}")
        
        # Domain analysis
        analysis.append(f"Full Domain: {parsed.netloc}")
        
        # Try to extract domain parts
        try:
            import tldextract
            extracted = tldextract.extract(url)
            
            if extracted.subdomain:
                subdomain_count = len(extracted.subdomain.split('.'))
                analysis.append(f"Subdomain: {extracted.subdomain} ({subdomain_count} level{'s' if subdomain_count > 1 else ''})")
                if subdomain_count > 2:
                    analysis.append("Multiple subdomain levels detected - sometimes used to create user confusion")
            
            analysis.append(f"Primary Domain: {extracted.domain}")
            analysis.append(f"Top-Level Domain: .{extracted.suffix}")
            
            # TLD analysis
            if extracted.suffix in self.suspicious_tlds:
                analysis.append(f"SUSPICIOUS TLD: .{extracted.suffix} is commonly associated with malicious websites and phishing campaigns")
            elif extracted.suffix in self.legitimate_tlds:
                analysis.append(f"Standard TLD: .{extracted.suffix} is a well-established, trustworthy domain extension")
            elif len(extracted.suffix) > 3:
                analysis.append(f"Long TLD: .{extracted.suffix} - newer domain extensions require additional verification")
                
        except ImportError:
            # Fallback domain analysis without tldextract
            domain_parts = parsed.netloc.split('.')
            if len(domain_parts) > 2:
                analysis.append(f"Subdomain Structure: {' → '.join(domain_parts)}")
            analysis.append(f"Top-Level Domain: .{domain_parts[-1]}")
        
        # Path analysis
        if parsed.path and parsed.path != '/':
            path_depth = len([p for p in parsed.path.split('/') if p])
            analysis.append(f"URL Path: {parsed.path} ({path_depth} level{'s' if path_depth > 1 else ''})")
            
            if path_depth > 4:
                analysis.append("Very deep directory structure - sometimes used to hide malicious content")
            elif path_depth > 2:
                analysis.append("Moderate directory depth - common for organized websites")
        else:
            analysis.append("URL Path: Root domain (/) - accessing main page")
        
        # Query parameters
        if parsed.query:
            param_count = len(parsed.query.split('&'))
            analysis.append(f"Query Parameters: {param_count} parameter{'s' if param_count > 1 else ''}")
            if len(parsed.query) > 100:
                analysis.append("Very long query string detected - may contain encoded malicious content or tracking data")
        
        # URL length analysis
        url_length = len(url)
        analysis.append(f"Total URL Length: {url_length} characters")
        if url_length > 200:
            analysis.append("EXTREMELY LONG URL: Excessive length is a strong phishing indicator")
        elif url_length > 100:
            analysis.append("Long URL: Above average length - monitor for suspicious patterns")
        elif url_length < 30:
            analysis.append("Concise URL: Short, clean URLs are generally more trustworthy")
        
        return analysis
    
    def _identify_risk_factors(self, url, prediction, confidence, features_dict):
        """Identify comprehensive risk factors."""
        risk_factors = []
        url_lower = url.lower()
        parsed = urlparse(url)
        
        print(f"DEBUG: Analyzing risk factors for URL: {url}")
        
        # Always show basic analysis
        risk_factors.append(f"Domain Analysis: {parsed.netloc}")
        risk_factors.append(f"URL Length: {len(url)} characters")
        risk_factors.append(f"Protocol: {parsed.scheme.upper()}")
        
        # Suspicious keyword analysis
        found_keywords = [kw for kw in self.suspicious_keywords if kw in url_lower]
        if found_keywords:
            risk_factors.append(f"SUSPICIOUS KEYWORDS: Found {len(found_keywords)} concerning terms: {', '.join(found_keywords[:5])}")
            risk_factors.append("These keywords are frequently used in phishing URLs to create urgency or mimic legitimate services")
        else:
            risk_factors.append("Keyword Analysis: No immediately suspicious keywords detected")
        
        # URL shortener detection
        if any(shortener in url_lower for shortener in self.url_shorteners):
            risk_factors.append("URL SHORTENER DETECTED: This URL uses a shortening service that can hide the true destination")
            risk_factors.append("Phishers often use URL shorteners to disguise malicious links and bypass security filters")
        
        # IP address detection
        if re.match(r'^\d+\.\d+\.\d+\.\d+', parsed.netloc):
            risk_factors.append("IP ADDRESS USAGE: URL connects directly to an IP address instead of a domain name")
            risk_factors.append("This is a major red flag - legitimate websites rarely use IP addresses directly")
        
        # Brand impersonation analysis
        brand_found = False
        for brand in self.brand_keywords:
            if brand in url_lower:
                if not (url_lower.startswith(f'https://{brand}.') or url_lower.startswith(f'https://www.{brand}.') or url_lower.startswith(f'http://{brand}.') or url_lower.startswith(f'http://www.{brand}.')):
                    risk_factors.append(f"POTENTIAL BRAND IMPERSONATION: URL contains '{brand}' but may not be the official website")
                    risk_factors.append(f"Verify this is actually {brand.title()}'s official domain before proceeding")
                    brand_found = True
                    break
        
        if not brand_found:
            risk_factors.append("Brand Analysis: No obvious brand impersonation detected")
        
        # Domain structure analysis
        dot_count = parsed.netloc.count('.')
        if dot_count > 3:
            risk_factors.append(f"COMPLEX DOMAIN STRUCTURE: {dot_count} dots in domain may indicate confusion tactics")
        elif dot_count > 2:
            risk_factors.append(f"Moderate Domain Complexity: {dot_count} dots detected")
        else:
            risk_factors.append(f"Simple Domain Structure: {dot_count} dots - standard format")
        
        # Length analysis with specific thresholds
        url_length = len(url)
        if url_length > 100:
            risk_factors.append(f"LONG URL DETECTED: {url_length} characters exceeds normal website standards")
        elif url_length > 75:
            risk_factors.append(f"Above Average Length: {url_length} characters - monitor for suspicious patterns")
        else:
            risk_factors.append(f"Normal Length: {url_length} characters - within expected range")
        
        # Path analysis
        if parsed.path and len(parsed.path) > 1:
            path_parts = [p for p in parsed.path.split('/') if p]
            if len(path_parts) > 3:
                risk_factors.append(f"Deep Path Structure: {len(path_parts)} directory levels detected")
            else:
                risk_factors.append(f"Standard Path: {len(path_parts)} directory levels")
        
        # HTTPS analysis
        if url.startswith('https://'):
            risk_factors.append("HTTPS Encryption: Secure protocol detected - positive indicator")
        else:
            risk_factors.append("NO HTTPS: URL uses HTTP instead of HTTPS - potential security risk")
            risk_factors.append("Modern legitimate websites should always use HTTPS for security")
        
        # High confidence analysis
        if prediction == 'phishing':
            if confidence > 0.95:
                risk_factors.append(f"VERY HIGH ML CONFIDENCE: {confidence*100:.1f}% confidence in phishing classification")
            elif confidence > 0.8:
                risk_factors.append(f"HIGH ML CONFIDENCE: {confidence*100:.1f}% confidence suggests strong phishing indicators")
            else:
                risk_factors.append(f"MODERATE ML CONFIDENCE: {confidence*100:.1f}% confidence in classification")
        else:
            risk_factors.append(f"LEGITIMATE CLASSIFICATION: {confidence*100:.1f}% confidence in legitimacy")
        
        print(f"DEBUG: Generated {len(risk_factors)} risk factors")
        return risk_factors
    
    def _get_security_recommendations(self, prediction, confidence):
        """Get security recommendations based on prediction."""
        if prediction == 'phishing':
            return [
                "DO NOT visit this URL or enter any personal information",
                "If received via email, mark as spam and delete the message",
                "Change passwords for any accounts you accessed recently",
                "Run a security scan on your device if you interacted with the site",
                "Report this URL to your IT department or security team"
            ]
        else:
            return [
                "URL appears safe, but always remain vigilant online",
                "Look for HTTPS encryption when entering sensitive information",
                "Verify the website's identity through official channels when in doubt",
                "Keep your browser and security software updated"
            ]
    
    def _get_technical_breakdown(self, features_dict, prediction, confidence):
        """Provide technical analysis details."""
        breakdown = []
        
        breakdown.extend([
            "MACHINE LEARNING ANALYSIS",
            f"Classification: {prediction.upper()}",
            f"Confidence Score: {confidence*100:.2f}%",
            f"Features Analyzed: {len(features_dict)}",
            f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        # Key feature highlights
        if 'url_length' in features_dict:
            breakdown.append(f"URL length feature: {features_dict['url_length']} characters")
        if 'num_dots' in features_dict:
            breakdown.append(f"Domain dots: {features_dict['num_dots']}")
        if 'https' in features_dict:
            breakdown.append(f"HTTPS: {'Yes' if features_dict['https'] == 1 else 'No'}")
        
        return breakdown
    
    def _analyze_advanced_url_intelligence(self, url, features_dict):
        """Advanced URL intelligence analysis for comprehensive level."""
        analysis = []
        parsed = urlparse(url)
        
        analysis.extend([
            "ADVANCED URL DECOMPOSITION",
            f"Full URL: {url}",
            f"Scheme: {parsed.scheme}",
            f"Network Location: {parsed.netloc}",
            f"Path: {parsed.path or '(root)'}",
            f"Query: {parsed.query or '(none)'}",
            f"Fragment: {parsed.fragment or '(none)'}",
            "",
            "DOMAIN HIERARCHY ANALYSIS",
            f"Total domain components: {len(parsed.netloc.split('.'))}",
            f"Domain breakdown: {' -> '.join(reversed(parsed.netloc.split('.')))}"
        ])
        
        # Check for potential typosquatting
        common_domains = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'paypal']
        for domain in common_domains:
            if domain in parsed.netloc.lower() and parsed.netloc.lower() != f"{domain}.com":
                analysis.append(f"TYPOSQUATTING ALERT: Domain resembles {domain}.com")
        
        # URL encoding analysis
        if '%' in url:
            analysis.append("URL ENCODING DETECTED: Contains percent-encoded characters")
            encoded_chars = len([c for c in url if c == '%'])
            analysis.append(f"Number of encoded sequences: {encoded_chars}")
        
        # Punycode detection
        if 'xn--' in url:
            analysis.append("PUNYCODE DETECTED: Domain uses international characters")
        
        # Path complexity analysis
        if parsed.path:
            path_segments = [seg for seg in parsed.path.split('/') if seg]
            analysis.extend([
                "",
                "PATH COMPLEXITY ANALYSIS",
                f"Path segments: {len(path_segments)}",
                f"Deepest directory level: {len(path_segments)}",
                f"Total path characters: {len(parsed.path)}"
            ])
            
            if len(path_segments) > 5:
                analysis.append("WARNING: Very deep path structure - unusual for legitimate sites")
        
        return analysis
    
    def _comprehensive_risk_assessment(self, url, prediction, confidence, features_dict):
        """Comprehensive risk assessment including domain age analysis."""
        risk_factors = []
        url_lower = url.lower()
        parsed = urlparse(url)
        
        # Start with comprehensive header
        risk_factors.extend([
            "COMPREHENSIVE THREAT ANALYSIS",
            f"Overall Risk Level: {'CRITICAL' if confidence > 0.9 and prediction == 'phishing' else 'HIGH' if confidence > 0.7 and prediction == 'phishing' else 'MODERATE' if prediction == 'phishing' else 'LOW'}",
            f"Confidence Score: {confidence*100:.2f}%",
            "",
            "DOMAIN AGE ANALYSIS"
        ])
        
        # Domain age analysis (enhanced)
        domain_age_analysis = self._analyze_domain_age(parsed.netloc)
        risk_factors.extend(domain_age_analysis)
        risk_factors.append("")
        
        # Advanced keyword analysis
        risk_factors.append("KEYWORD THREAT ANALYSIS")
        high_risk_keywords = ['login', 'signin', 'verify', 'urgent', 'suspended', 'update']
        medium_risk_keywords = ['secure', 'account', 'confirm', 'click', 'offer', 'free']
        
        found_high_risk = [kw for kw in high_risk_keywords if kw in url_lower]
        found_medium_risk = [kw for kw in medium_risk_keywords if kw in url_lower]
        
        if found_high_risk:
            risk_factors.append(f"HIGH-RISK keywords detected: {', '.join(found_high_risk)}")
        if found_medium_risk:
            risk_factors.append(f"MEDIUM-RISK keywords detected: {', '.join(found_medium_risk)}")
        if not found_high_risk and not found_medium_risk:
            risk_factors.append("No obvious malicious keywords detected")
        
        risk_factors.append("")
        
        # Brand impersonation deep analysis
        risk_factors.append("BRAND IMPERSONATION ANALYSIS")
        brand_risks = self._deep_brand_analysis(url_lower, parsed.netloc)
        risk_factors.extend(brand_risks)
        risk_factors.append("")
        
        # Infrastructure analysis
        risk_factors.append("INFRASTRUCTURE ANALYSIS")
        if re.match(r'^\d+\.\d+\.\d+\.\d+', parsed.netloc):
            risk_factors.append("CRITICAL: Direct IP address usage - major red flag")
        else:
            risk_factors.append(f"Domain-based URL: {parsed.netloc}")
        
        # CDN/Cloud service detection
        cloud_indicators = ['amazonaws', 'cloudfront', 'azurewebsites', 'heroku', 'github.io']
        if any(indicator in url_lower for indicator in cloud_indicators):
            risk_factors.append("Cloud service hosting detected - verify legitimacy")
        
        # URL length risk analysis
        risk_factors.append("")
        risk_factors.append("URL LENGTH RISK ANALYSIS")
        length_risk = self._analyze_length_risk(len(url))
        risk_factors.extend(length_risk)
        
        return risk_factors
    
    def _analyze_domain_age(self, domain):
        """Analyze domain age (simulated - would use WHOIS in real implementation)."""
        analysis = []
        
        # Check for new TLDs (often used by attackers)
        tld = domain.split('.')[-1]
        if tld in ['tk', 'ml', 'ga', 'cf']:
            analysis.extend([
                f"Domain TLD: .{tld} (Free/Suspicious TLD)",
                "WARNING: Domain uses TLD commonly associated with temporary/malicious sites",
                "ESTIMATED AGE: Likely very recent (high risk indicator)"
            ])
        elif tld in ['com', 'org', 'net']:
            analysis.extend([
                f"Domain TLD: .{tld} (Standard TLD)",
                "Standard top-level domain - requires verification",
                "AGE ANALYSIS: Unable to determine exact age without WHOIS query"
            ])
        else:
            analysis.extend([
                f"Domain TLD: .{tld}",
                "Country-code or newer TLD - verify legitimacy",
                "RECOMMENDATION: Check domain registration date manually"
            ])
        
        # Domain name pattern analysis for age estimation
        if len(domain.replace('.', '')) < 6:
            analysis.append("SHORT DOMAIN: May indicate premium/aged domain OR typosquatting")
        elif any(char.isdigit() for char in domain):
            analysis.append("NUMERIC CHARACTERS: Often indicates newer/generated domain")
        
        # Subdomain age indicators
        if domain.count('.') > 2:
            analysis.append("MULTIPLE SUBDOMAINS: May indicate recent creation or compromise")
        
        analysis.extend([
            "",
            "DOMAIN AGE RECOMMENDATIONS",
            "Manually verify domain registration date using WHOIS lookup",
            "Domains less than 30 days old are higher risk",
            "Legitimate businesses typically use established domains",
            "Be extra cautious with domains registered recently"
        ])
        
        return analysis
    
    def _deep_brand_analysis(self, url_lower, netloc):
        """Deep brand impersonation analysis."""
        analysis = []
        
        major_brands = {
            'paypal': 'paypal.com',
            'amazon': 'amazon.com',
            'microsoft': 'microsoft.com',
            'google': 'google.com',
            'apple': 'apple.com',
            'facebook': 'facebook.com',
            'bank': 'various banking institutions'
        }
        
        found_impersonation = False
        for brand, official_domain in major_brands.items():
            if brand in url_lower:
                if not (netloc == official_domain or netloc == f"www.{official_domain}"):
                    analysis.extend([
                        f"BRAND IMPERSONATION DETECTED: {brand.upper()}",
                        f"Official domain should be: {official_domain}",
                        f"Actual domain: {netloc}",
                        "RISK: HIGH - Likely phishing attempt"
                    ])
                    found_impersonation = True
                else:
                    analysis.append(f"LEGITIMATE {brand.upper()} domain verified")
        
        if not found_impersonation:
            # Check for character substitution attacks
            suspicious_chars = ['0', '1', 'rn', 'vv', 'nn']
            if any(char in netloc for char in suspicious_chars):
                analysis.append("POSSIBLE CHARACTER SUBSTITUTION: Domain may use look-alike characters")
            else:
                analysis.append("No obvious brand impersonation detected")
        
        return analysis
    
    def _analyze_length_risk(self, url_length):
        """Analyze URL length risk in detail."""
        analysis = []
        
        if url_length > 150:
            analysis.extend([
                f"CRITICAL LENGTH: {url_length} characters",
                "URLs over 150 characters are extremely suspicious",
                "RISK LEVEL: CRITICAL"
            ])
        elif url_length > 100:
            analysis.extend([
                f"HIGH LENGTH: {url_length} characters", 
                "URLs over 100 characters often indicate phishing",
                "RISK LEVEL: HIGH"
            ])
        elif url_length > 75:
            analysis.extend([
                f"MODERATE LENGTH: {url_length} characters",
                "Above average length - monitor for other indicators", 
                "RISK LEVEL: MODERATE"
            ])
        else:
            analysis.extend([
                f"NORMAL LENGTH: {url_length} characters",
                "URL length within expected range",
                "LENGTH RISK: LOW"
            ])
        
        return analysis
    
    def _analyze_security_posture(self, url, features_dict, prediction):
        """Comprehensive security posture analysis."""
        analysis = []
        security_score = 0
        max_possible = 100
        
        analysis.extend([
            "COMPREHENSIVE SECURITY POSTURE EVALUATION",
            "",
            "HTTPS ENCRYPTION ANALYSIS"
        ])
        
        # HTTPS Analysis (25 points)
        if url.startswith('https://'):
            security_score += 25
            analysis.extend([
                "STATUS: ENABLED (+25 points)",
                "Secure data transmission",
                "SSL/TLS certificate present",
                "Connection encrypted"
            ])
        else:
            analysis.extend([
                "STATUS: MISSING (0 points)",
                "CRITICAL: Data transmitted in clear text", 
                "No SSL/TLS protection",
                "Vulnerable to interception"
            ])
        
        analysis.append("")
        analysis.append("DOMAIN STRUCTURE ANALYSIS")
        
        # Domain Structure (20 points)
        parsed = urlparse(url)
        dot_count = parsed.netloc.count('.')
        if dot_count <= 2:
            security_score += 20
            analysis.append("STRUCTURE: CLEAN (+20 points)")
        elif dot_count <= 4:
            security_score += 10
            analysis.append("STRUCTURE: MODERATE (+10 points)")
        else:
            analysis.append("STRUCTURE: COMPLEX (0 points)")
        
        analysis.append("")
        analysis.append("URL LENGTH ANALYSIS")
        
        # URL Length (15 points)
        url_length = len(url)
        if url_length < 50:
            security_score += 15
            analysis.append("LENGTH: OPTIMAL (+15 points)")
        elif url_length < 100:
            security_score += 10
            analysis.append("LENGTH: ACCEPTABLE (+10 points)")
        else:
            analysis.append("LENGTH: EXCESSIVE (0 points)")
        
        analysis.append("")
        analysis.append("PROTOCOL SECURITY ANALYSIS")
        
        # Protocol Security (15 points)
        if parsed.scheme == 'https':
            security_score += 15
            analysis.append("PROTOCOL: SECURE (+15 points)")
        else:
            analysis.append("PROTOCOL: INSECURE (0 points)")
        
        analysis.append("")
        analysis.append("ML CLASSIFICATION ANALYSIS")
        
        # ML Prediction Alignment (25 points)
        if prediction == 'legitimate':
            security_score += 25
            analysis.append("CLASSIFICATION: LEGITIMATE (+25 points)")
        else:
            analysis.append("CLASSIFICATION: PHISHING (0 points)")
        
        final_score = min(security_score, max_possible)
        analysis.extend([
            "",
            f"FINAL SECURITY SCORE: {final_score}/100",
            ""
        ])
        
        if final_score >= 90:
            analysis.append("RATING: EXCELLENT - Comprehensive security measures in place")
        elif final_score >= 70:
            analysis.append("RATING: GOOD - Solid security with minor improvements needed")
        elif final_score >= 50:
            analysis.append("RATING: CONCERNING - Multiple security issues identified")
        else:
            analysis.append("RATING: CRITICAL - Severe security deficiencies detected")
        
        return analysis
    
    def _ml_deep_analysis(self, features_dict, prediction, confidence):
        """Deep machine learning analysis for comprehensive level."""
        analysis = []
        
        analysis.extend([
            "MACHINE LEARNING DEEP DIVE ANALYSIS",
            "",
            "MODEL PERFORMANCE METRICS",
            f"Primary Classification: {prediction.upper()}",
            f"Confidence Score: {confidence*100:.3f}%", 
            f"Prediction Certainty: {abs(confidence - 0.5) * 200:.1f}%",
            f"Feature Vector Dimension: {len(features_dict)}",
            ""
        ])
        
        # Feature importance simulation
        analysis.append("TOP CONTRIBUTING FEATURES")
        feature_impacts = []
        
        if features_dict.get('url_length', 0) > 75:
            feature_impacts.append(("URL Length", "HIGH IMPACT", features_dict.get('url_length', 0)))
        if features_dict.get('num_dots', 0) > 3:
            feature_impacts.append(("Domain Dots", "HIGH IMPACT", features_dict.get('num_dots', 0)))
        if features_dict.get('https', 0) == 0:
            feature_impacts.append(("HTTPS Missing", "HIGH IMPACT", "No"))
        
        for feature, impact, value in feature_impacts[:5]:
            analysis.append(f"{feature}: {impact} (Value: {value})")
        
        # Model confidence analysis
        analysis.extend([
            "",
            "CONFIDENCE ANALYSIS",
            f"Decision Boundary Distance: {abs(confidence - 0.5):.3f}",
            f"Classification Strength: {'Very Strong' if confidence > 0.9 else 'Strong' if confidence > 0.8 else 'Moderate' if confidence > 0.7 else 'Weak'}",
            f"Prediction Reliability: {'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}"
        ])
        
        # Feature distribution analysis
        numeric_features = [v for v in features_dict.values() if isinstance(v, (int, float)) and v != -1]
        if numeric_features:
            analysis.extend([
                "",
                "FEATURE STATISTICS",
                f"Mean Feature Value: {np.mean(numeric_features):.3f}",
                f"Feature Standard Deviation: {np.std(numeric_features):.3f}",
                f"Feature Range: {np.min(numeric_features):.2f} to {np.max(numeric_features):.2f}",
                f"Active Features: {sum(1 for v in numeric_features if v > 0)}/{len(numeric_features)}"
            ])
        
        return analysis
    
    def _threat_intelligence_analysis(self, url, prediction, confidence):
        """Threat intelligence and contextual analysis."""
        analysis = []
        parsed = urlparse(url)
        
        analysis.extend([
            "THREAT INTELLIGENCE & CONTEXTUAL ANALYSIS",
            "",
            "THREAT LANDSCAPE CONTEXT",
            f"Current Threat Level: {'CRITICAL' if prediction == 'phishing' and confidence > 0.9 else 'ELEVATED' if prediction == 'phishing' else 'LOW'}",
            f"Attack Vector: Web-based phishing",
            f"Primary Target: User credentials and personal information"
        ])
        
        # Analyze attack patterns
        if prediction == 'phishing':
            analysis.extend([
                "",
                "ATTACK PATTERN ANALYSIS",
                "Type: Credential harvesting attempt",
                "Method: Social engineering via deceptive URL",
                "Goal: Identity theft and financial fraud"
            ])
            
            # Check for common phishing patterns
            if 'login' in url.lower():
                analysis.append("Pattern: Login page impersonation")
            if any(brand in url.lower() for brand in ['bank', 'paypal', 'amazon']):
                analysis.append("Pattern: Brand impersonation")
        
        # Geographic and infrastructure context
        analysis.extend([
            "",
            "INFRASTRUCTURE INTELLIGENCE",
            f"Domain: {parsed.netloc}",
            f"Protocol Security: {'Encrypted' if parsed.scheme == 'https' else 'Unencrypted'}",
            "Infrastructure Analysis: Standard web hosting"
        ])
        
        # Temporal context
        analysis.extend([
            "",
            "TEMPORAL ANALYSIS",
            f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Threat Persistence: Unknown (single-point analysis)",
            "Recommendation: Monitor for pattern changes"
        ])
        
        return analysis
    
    def _comprehensive_security_recommendations(self, prediction, confidence, url):
        """Comprehensive security recommendations based on analysis."""
        recommendations = []
        
        if prediction == 'phishing':
            recommendations.extend([
                "IMMEDIATE CRITICAL ACTIONS",
                "DO NOT PROCEED - This URL poses significant security risks",
                "DO NOT enter any personal information, passwords, or payment details",
                "DO NOT download any files or software from this site",
                "CLOSE the browser tab immediately",
                "",
                "INCIDENT RESPONSE PROCEDURES",
                "If you already visited the site:",
                "Change all passwords immediately, starting with most critical accounts",
                "Enable two-factor authentication on all accounts",
                "Run comprehensive malware scan on your device",
                "Monitor bank statements and credit reports closely",
                "",
                "If you entered information:",
                "Contact your bank/financial institutions immediately",
                "Place fraud alerts on your credit accounts", 
                "Consider credit monitoring services",
                "Document the incident for potential reporting",
                "",
                "ORGANIZATIONAL RESPONSE",
                "Report to IT Security team immediately",
                "Forward phishing email to security@[organization].com",
                "Update security awareness training based on this threat",
                "Consider blocking this domain at network level",
                "",
                "PREVENTIVE MEASURES",
                "Implement email filtering to block similar threats",
                "Educate users about this specific attack pattern",
                "Review and update security policies",
                "Conduct phishing simulation training"
            ])
        else:
            recommendations.extend([
                "SECURITY BEST PRACTICES FOR LEGITIMATE SITES",
                "This URL appears legitimate, but maintain security vigilance",
                "",
                "VERIFICATION STEPS",
                "Verify SSL certificate details before entering sensitive information",
                "Confirm the website URL matches exactly what you expect",
                "Look for trust indicators (security badges, contact information)",
                "Cross-reference with official company communications",
                "",
                "ONGOING SECURITY MEASURES",
                "Use unique, complex passwords for each account",
                "Enable two-factor authentication where available",
                "Keep browsers and security software updated",
                "Use reputable password managers",
                "Regularly monitor account activity",
                "",
                "ORGANIZATIONAL BEST PRACTICES",
                "Maintain whitelist of approved business websites",
                "Implement web filtering and security monitoring",
                "Provide regular security awareness training",
                "Establish clear protocols for accessing business applications"
            ])
            
            if confidence < 0.8:
                recommendations.extend([
                    "",
                    "ADDITIONAL CAUTION (Moderate Confidence)",
                    "Exercise heightened vigilance due to mixed classification signals",
                    "Verify website authenticity through independent channels",
                    "Consider using alternative official channels for sensitive transactions",
                    "Monitor for unusual website behavior or requests"
                ])
        
        return recommendations
    
    def _technical_deep_dive(self, features_dict, url, prediction):
        """Comprehensive technical deep dive analysis."""
        analysis = []
        parsed = urlparse(url)
        
        analysis.extend([
            "TECHNICAL DEEP DIVE ANALYSIS",
            "",
            "URL PARSING AND STRUCTURE",
            f"Complete URL: {url}",
            f"Scheme: {parsed.scheme}",
            f"Network Location: {parsed.netloc}",
            f"Path: {parsed.path or '(none)'}",
            f"Query String: {parsed.query or '(none)'}",
            f"Fragment: {parsed.fragment or '(none)'}",
            ""
        ])
        
        # Character encoding analysis
        analysis.append("CHARACTER ENCODING ANALYSIS")
        if any(ord(char) > 127 for char in url):
            analysis.append("Contains non-ASCII characters - potential internationalization")
        else:
            analysis.append("Pure ASCII encoding - standard format")
        
        if '%' in url:
            analysis.append("URL encoding detected - some characters are percent-encoded")
        
        # Protocol analysis
        analysis.extend([
            "",
            "PROTOCOL ANALYSIS",
            f"Transport Protocol: {parsed.scheme.upper()}",
            f"Default Port: {443 if parsed.scheme == 'https' else 80}",
            f"Security Layer: {'TLS/SSL' if parsed.scheme == 'https' else 'None'}"
        ])
        
        # Feature vector analysis
        analysis.extend([
            "",
            "MACHINE LEARNING FEATURE VECTOR",
            f"Total Features: {len(features_dict)}",
            f"Non-zero Features: {sum(1 for v in features_dict.values() if v != 0)}",
            f"Binary Features: {sum(1 for v in features_dict.values() if v in [0, 1])}",
            f"Continuous Features: {sum(1 for v in features_dict.values() if isinstance(v, float) and v not in [0, 1])}"
        ])
        
        # Key feature breakdown
        if features_dict:
            analysis.append("")
            analysis.append("KEY FEATURE VALUES")
            for feature, value in sorted(features_dict.items())[:10]:
                feature_display = feature.replace('_', ' ').title()
                analysis.append(f"{feature_display}: {value}")
        
        # Classification boundary analysis
        analysis.extend([
            "",
            "CLASSIFICATION ANALYSIS",
            f"Decision Boundary: 0.5 (standard threshold)",
            f"Actual Score: {features_dict.get('prediction_score', 'N/A')}",
            f"Classification: {prediction.upper()}",
            f"Margin: Distance from decision boundary"
        ])
        
        return analysis
    
    def _get_feature_importance(self, features_dict):
        """Get feature importance scores (simplified version)."""
        importance_scores = {}
        
        # Assign importance based on feature values and known risk factors
        for feature, value in features_dict.items():
            if 'length' in feature and value > 75:
                importance_scores[feature] = 0.3
            elif 'dots' in feature and value > 3:
                importance_scores[feature] = 0.25
            elif 'https' in feature:
                importance_scores[feature] = 0.2 if value == 0 else -0.1
            elif 'special' in feature and value > 8:
                importance_scores[feature] = 0.15
            else:
                importance_scores[feature] = abs(value) * 0.01  # Small default importance
        
        return importance_scores