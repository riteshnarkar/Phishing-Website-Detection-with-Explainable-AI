"""
Complete Flask App with Enhanced Explanation Levels
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from datetime import datetime
from predictor import PhishingPredictor  # Your enhanced predictor
import threading
import time
import traceback

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize predictor (global variable)
predictor = None
predictor_lock = threading.Lock()

def initialize_predictor():
    """Initialize the predictor in a thread-safe manner."""
    global predictor
    with predictor_lock:
        if predictor is None:
            try:
                predictor = PhishingPredictor()
                print("Predictor initialized successfully")
            except Exception as e:
                print(f"Error initializing predictor: {str(e)}")
                predictor = None

@app.before_request
def startup():
    """Initialize predictor before first request."""
    initialize_predictor()

@app.route('/')
def index():
    """Main page with enhanced explanation options."""
    return render_template('index.html')

# MAIN ROUTE - Updated to support explanation levels
@app.route('/analyze', methods=['POST'])
def analyze_url():
    """Analyze a single URL and return results with enhanced explanations."""
    global predictor
    
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Predictor not available. Please check if predictor.py exists and is properly configured.'
        }), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
            
        url = data.get('url', '').strip()
        model_name = data.get('model', 'xgboost')
        include_explanation = data.get('explain', True)
        
        # CRITICAL: Get explanation level from request
        explanation_level = data.get('explanation_level', 'detailed')
        print(f"DEBUG: Received explanation_level: {explanation_level}")
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # ENHANCED: Pass explanation level to predictor
        result = predictor.predict_url(url, model_name, include_explanation, explanation_level)
        
        print(f"DEBUG: Prediction result keys: {list(result.keys())}")
        if 'detailed_explanation' in result:
            print(f"DEBUG: Detailed explanation sections: {len(result['detailed_explanation'].get('sections', []))}")
        
        # Format response
        response = {
            'success': True,
            'url': result['url'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'risk_level': result['risk_level'],
            'phishing_probability': result['phishing_probability'],
            'model_used': result['model_used'],
            'timestamp': result['timestamp'],
            'explanation_level': explanation_level
        }
        
        # FIXED: Better explanation handling
        if include_explanation:
            if 'detailed_explanation' in result and result['detailed_explanation']:
                # Enhanced explanation format
                response['explanation'] = result['detailed_explanation']
                print(f"DEBUG: Sending detailed explanation with {len(result['detailed_explanation'].get('sections', []))} sections")
            elif 'explanation' in result:
                # Fallback to basic explanation
                response['explanation'] = result['explanation']
                print(f"DEBUG: Sending basic explanation: {result['explanation'][:100]}...")
            else:
                # No explanation available
                response['explanation'] = f"Analysis completed. URL classified as {result['prediction']} with {result['confidence']:.1%} confidence."
                print("DEBUG: No explanation found, using fallback")
        
        print(f"DEBUG: Final response explanation type: {type(response.get('explanation'))}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Analysis failed: {str(e)}'
        print(f"ERROR in analyze_url: {error_msg}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': error_msg}), 500

# NEW ROUTE: Quick analysis with basic explanation
@app.route('/analyze-quick', methods=['POST'])
def analyze_quick():
    """
    Quick analysis with basic explanation (original 3-line format).
    
    Expected JSON: {"url": "https://example.com"}
    """
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # USE: Basic explanation level
        result = predictor.predict_url(url, 'xgboost', True, 'basic')
        
        return jsonify({
            'success': True,
            'url': result['url'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'explanation': result.get('explanation', 'No explanation available'),
            'timestamp': result['timestamp']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW ROUTE: Detailed analysis
@app.route('/analyze-detailed', methods=['POST'])
def analyze_detailed():
    """
    Detailed analysis with multiple explanation sections.
    
    Expected JSON: {"url": "https://example.com", "model": "xgboost"}
    """
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        model_name = data.get('model', 'xgboost')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # USE: Detailed explanation level
        result = predictor.predict_url(url, model_name, True, 'detailed')
        
        return jsonify({
            'success': True,
            'analysis': result,
            'explanation_level': 'detailed'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW ROUTE: Comprehensive analysis with statistics
@app.route('/analyze-comprehensive', methods=['POST'])
def analyze_comprehensive():
    """
    Comprehensive analysis with all details and statistics.
    
    Expected JSON: {"url": "https://example.com", "model": "xgboost"}
    """
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        model_name = data.get('model', 'xgboost')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # USE: Maximum detail with statistical analysis
        result = predictor.analyze_url_comprehensive(url, model_name)
        
        return jsonify({
            'success': True,
            'comprehensive_analysis': result,
            'explanation_level': 'comprehensive_plus',
            'includes_statistics': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# UPDATED: Batch analysis with explanation levels
@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple URLs with configurable explanation levels.
    
    Expected JSON:
    {
        "urls": ["url1", "url2", ...],
        "model": "xgboost",
        "explain": true,
        "explanation_level": "basic"  // NEW: to avoid performance issues with many URLs
    }
    """
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        model_name = data.get('model', 'xgboost')
        include_explanations = data.get('explain', False)
        explanation_level = data.get('explanation_level', 'basic')  # Default to basic for performance
        
        if not urls:
            return jsonify({'error': 'URLs list is required'}), 400
        
        # Limit batch size
        if len(urls) > 100:
            return jsonify({'error': 'Maximum 100 URLs allowed per batch'}), 400
        
        # Clean URLs
        cleaned_urls = []
        for url in urls:
            url = url.strip()
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
                cleaned_urls.append(url)
        
        # USE: Batch analysis with explanation level
        results = predictor.predict_batch(cleaned_urls, model_name, include_explanations, explanation_level)
        
        # Calculate summary statistics
        total = len(results)
        phishing_count = sum(1 for r in results if r.get('prediction') == 'phishing')
        legitimate_count = sum(1 for r in results if r.get('prediction') == 'legitimate')
        error_count = sum(1 for r in results if r.get('prediction') == 'error')
        
        summary = {
            'total': total,
            'phishing': phishing_count,
            'legitimate': legitimate_count,
            'errors': error_count,
            'phishing_percentage': (phishing_count / total * 100) if total > 0 else 0
        }
        
        response = {
            'success': True,
            'results': results,
            'summary': summary,
            'model_used': model_name,
            'explanation_level': explanation_level,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Batch analysis failed: {str(e)}'
        print(f"Error in batch_analyze: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

# ENHANCED: Demo page with explanation level options
@app.route('/demo')
def demo_page():
    """Enhanced demo page with explanation level selection."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Phishing Detection Demo</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 900px; margin: 0 auto; background: white;
                border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 30px; text-align: center;
            }
            .content { padding: 30px; }
            .form-group { margin-bottom: 20px; }
            .form-group label { display: block; margin-bottom: 8px; font-weight: 600; }
            input[type="text"], select { 
                width: 100%; padding: 12px; border: 2px solid #e1e5e9;
                border-radius: 8px; font-size: 16px;
            }
            .explanation-level {
                display: grid; grid-template-columns: 1fr 1fr 1fr;
                gap: 15px; margin: 20px 0;
            }
            .level-option {
                padding: 15px; border: 2px solid #e1e5e9; border-radius: 8px;
                text-align: center; cursor: pointer; transition: all 0.3s;
            }
            .level-option.selected {
                border-color: #667eea; background: #f0f4ff;
            }
            .level-option h4 { margin: 0 0 8px 0; color: #333; }
            .level-option p { margin: 0; font-size: 0.9em; color: #666; }
            button { 
                width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; border: none; border-radius: 8px; cursor: pointer; 
                font-size: 16px; font-weight: 600;
            }
            .result { margin-top: 30px; }
            .explanation-section {
                margin: 15px 0; padding: 20px; background: #f8f9fa;
                border-radius: 8px; border-left: 4px solid #667eea;
            }
            .explanation-section h4 { margin: 0 0 15px 0; color: #333; }
            .explanation-section ul { margin: 0; padding-left: 20px; }
            .explanation-section li { margin: 8px 0; line-height: 1.4; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ Enhanced Phishing Detection</h1>
                <p>Choose your explanation level for detailed analysis</p>
            </div>
            
            <div class="content">
                <div class="form-group">
                    <label for="url">Enter URL to analyze:</label>
                    <input type="text" id="url" placeholder="https://example.com or suspicious-site.com" required>
                </div>
                
                <div class="form-group">
                    <label>Select Explanation Level:</label>
                    <div class="explanation-level">
                        <div class="level-option" onclick="selectLevel('basic')" id="basic">
                            <h4>🚀 Quick</h4>
                            <p>Basic 3-line explanation<br>Fast results</p>
                        </div>
                        <div class="level-option selected" onclick="selectLevel('detailed')" id="detailed">
                            <h4>🔍 Detailed</h4>
                            <p>Multiple sections<br>Comprehensive analysis</p>
                        </div>
                        <div class="level-option" onclick="selectLevel('comprehensive')" id="comprehensive">
                            <h4>🔬 Complete</h4>
                            <p>Full analysis + statistics<br>Maximum detail</p>
                        </div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="model">ML Model:</label>
                    <select id="model">
                        <option value="xgboost">XGBoost</option>
                        <option value="random_forest">Random Forest</option>
                        <option value="neural_network">Neural Network</option>
                    </select>
                </div>
                
                <button onclick="analyzeURL()" id="analyze-btn">🔍 Analyze URL</button>
                
                <div id="result"></div>
            </div>
        </div>

        <script>
            let selectedLevel = 'detailed';
            
            function selectLevel(level) {
                // Remove selection from all options
                document.querySelectorAll('.level-option').forEach(el => {
                    el.classList.remove('selected');
                });
                
                // Select current option
                document.getElementById(level).classList.add('selected');
                selectedLevel = level;
            }
            
            async function analyzeURL() {
                const url = document.getElementById('url').value.trim();
                const model = document.getElementById('model').value;
                const resultDiv = document.getElementById('result');
                const analyzeBtn = document.getElementById('analyze-btn');
                
                if (!url) {
                    alert('Please enter a URL');
                    return;
                }
                
                // Show loading
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = '🔄 Analyzing...';
                resultDiv.innerHTML = '<div style="text-align: center; padding: 40px; color: #666;">🔍 Analyzing URL, please wait...</div>';
                
                try {
                    // USAGE: Send explanation level to backend
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            url: url,
                            model: model,
                            explain: true,
                            explanation_level: selectedLevel  // This determines the detail level
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        displayResults(data);
                    } else {
                        displayError(data.error);
                    }
                } catch (error) {
                    displayError('Request failed: ' + error.message);
                } finally {
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = '🔍 Analyze URL';
                }
            }
            
            function displayResults(data) {
                const resultDiv = document.getElementById('result');
                const confidence = Math.round(data.confidence * 100);
                const predictionClass = data.prediction;
                
                let html = `
                    <div style="border-radius: 8px; overflow: hidden; margin-top: 20px;">
                        <div style="padding: 20px; color: white; background: ${predictionClass === 'phishing' ? 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)' : 'linear-gradient(135deg, #51cf66 0%, #40c057 100%)'};">
                            <h3 style="margin: 0;">${data.prediction.toUpperCase()}</h3>
                            <p style="margin: 5px 0 0 0;">Confidence: ${confidence}% | Risk: ${data.risk_level.replace('_', ' ')}</p>
                        </div>
                        
                        <div style="padding: 20px; background: #f8f9fa;">
                            <p><strong>🌐 URL:</strong> <code style="background: #e9ecef; padding: 4px 8px; border-radius: 4px;">${data.url}</code></p>
                            <p><strong>🤖 Model:</strong> ${data.model_used} | <strong>📊 Level:</strong> ${data.explanation_level}</p>
                `;
                
                // Display explanations based on what's available
                if (data.explanation && data.explanation.sections && data.explanation.sections.length > 0) {
                    // Enhanced explanation with sections
                    html += `<div style="margin-top: 20px;">`;
                    html += `<div style="padding: 15px; background: white; border-radius: 8px; margin-bottom: 15px;">`;
                    html += `<h4 style="margin: 0 0 10px 0; color: #333;">📝 Summary</h4>`;
                    html += `<p style="margin: 0; line-height: 1.5;">${data.explanation.summary}</p>`;
                    html += `</div>`;
                    
                    data.explanation.sections.forEach(section => {
                        html += `
                            <div class="explanation-section">
                                <h4>${section.title}</h4>
                                <ul style="margin: 0; padding-left: 20px;">
                        `;
                        
                        section.content.forEach(item => {
                            if (item.trim()) {  // Skip empty items
                                html += `<li style="margin: 8px 0; line-height: 1.4;">${item}</li>`;
                            }
                        });
                        
                        html += `</ul></div>`;
                    });
                    
                    html += `</div>`;
                } else if (data.explanation && typeof data.explanation === 'string') {
                    // Basic explanation
                    html += `
                        <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;">
                            <h4 style="margin: 0 0 10px 0;">📝 Analysis Explanation</h4>
                            <p style="margin: 0; line-height: 1.5;">${data.explanation}</p>
                        </div>
                    `;
                }
                
                html += `</div></div>`;
                resultDiv.innerHTML = html;
            }
            
            function displayError(errorMessage) {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `
                    <div style="background: #ffe0e0; color: #d63031; padding: 15px; border-radius: 8px; border-left: 4px solid #d63031; margin-top: 20px;">
                        <strong>❌ Error:</strong> ${errorMessage}
                    </div>
                `;
            }
            
            // Enter key support
            document.getElementById('url').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    analyzeURL();
                }
            });
        </script>
    </body>
    </html>
    '''

# EXAMPLE: Frontend JavaScript for different use cases
@app.route('/examples')
def api_examples():
    """Show API usage examples."""
    return jsonify({
        'api_endpoints': {
            '/analyze': {
                'description': 'Main analysis with configurable explanation level',
                'method': 'POST',
                'payload_example': {
                    'url': 'https://example.com',
                    'model': 'xgboost',
                    'explain': True,
                    'explanation_level': 'comprehensive'  # basic/detailed/comprehensive
                },
                'use_case': 'Default route - supports all explanation levels'
            },
            '/analyze-quick': {
                'description': 'Quick analysis with basic explanation',
                'method': 'POST',
                'payload_example': {
                    'url': 'https://example.com'
                },
                'use_case': 'When you need fast results with minimal explanation'
            },
            '/analyze-detailed': {
                'description': 'Detailed analysis with multiple sections',
                'method': 'POST',
                'payload_example': {
                    'url': 'https://example.com',
                    'model': 'xgboost'
                },
                'use_case': 'When you need comprehensive but not statistical analysis'
            },
            '/analyze-comprehensive': {
                'description': 'Maximum detail with statistics',
                'method': 'POST',
                'payload_example': {
                    'url': 'https://example.com',
                    'model': 'xgboost'
                },
                'use_case': 'For security research or detailed forensic analysis'
            },
            '/batch-analyze': {
                'description': 'Batch analysis with explanation level control',
                'method': 'POST',
                'payload_example': {
                    'urls': ['https://site1.com', 'https://site2.com'],
                    'model': 'xgboost',
                    'explain': True,
                    'explanation_level': 'basic'  # Recommended for batch to avoid timeout
                },
                'use_case': 'Analyze multiple URLs efficiently'
            }
        },
        'javascript_examples': {
            'quick_analysis': '''
// Quick analysis (original 3-line explanation)
fetch('/analyze-quick', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url: 'https://suspicious-site.com'})
})
.then(response => response.json())
.then(data => {
    console.log('Basic explanation:', data.explanation);
});
            ''',
            'detailed_analysis': '''
// Detailed analysis (multiple sections)
fetch('/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        url: 'https://suspicious-site.com',
        explanation_level: 'detailed'
    })
})
.then(response => response.json())
.then(data => {
    console.log('Summary:', data.explanation.summary);
    data.explanation.sections.forEach(section => {
        console.log(section.title, section.content);
    });
});
            ''',
            'comprehensive_analysis': '''
// Comprehensive analysis (all sections + statistics)
fetch('/analyze-comprehensive', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        url: 'https://suspicious-site.com'
    })
})
.then(response => response.json())
.then(data => {
    const analysis = data.comprehensive_analysis;
    console.log('Full analysis:', analysis.detailed_explanation);
    console.log('Statistics:', analysis.statistical_analysis);
});
            '''
        }
    })

# Rest of your existing routes remain the same...
@app.route('/models')
def get_models():
    """Get available models information."""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        model_info = predictor.get_model_info()
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    global predictor
    
    status = {
        'status': 'healthy' if predictor is not None else 'unhealthy',
        'predictor_initialized': predictor is not None,
        'explanation_levels_available': ['basic', 'detailed', 'comprehensive'],
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status)

@app.route('/api/v1/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions with explanation level support."""
    global predictor
    
    if predictor is None:
        return jsonify({'error': 'Service unavailable'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url'].strip()
        model_name = data.get('model', 'xgboost')
        include_explanation = data.get('explain', True)
        explanation_level = data.get('explanation_level', 'basic')  # API defaults to basic for compatibility
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # USE: API with explanation level
        result = predictor.predict_url(url, model_name, include_explanation, explanation_level)
        
        # API response format
        api_response = {
            'url': result['url'],
            'prediction': result['prediction'],
            'confidence': round(result['confidence'], 4),
            'risk_level': result['risk_level'],
            'probability': round(result['phishing_probability'], 4),
            'model': result['model_used'],
            'explanation_level': explanation_level,
            'timestamp': result['timestamp']
        }
        
        if include_explanation:
            if 'detailed_explanation' in result:
                api_response['explanation'] = result['detailed_explanation']
            else:
                api_response['explanation'] = result.get('explanation', 'No explanation available')
        
        return jsonify(api_response)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in api_predict: {error_msg}")
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    print("Starting Enhanced Flask Phishing Detection App...")
    print("Available explanation levels: basic, detailed, comprehensive")
    print("Visit /demo for interactive interface")
    print("Visit /examples for API usage examples")
    
    # For development only
    app.run(debug=True, host='0.0.0.0', port=5000)