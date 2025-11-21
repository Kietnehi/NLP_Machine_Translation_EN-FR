"""
Flask Web App for English to French Translation
"""
from flask import Flask, render_template, request, jsonify
import torch
import warnings
warnings.filterwarnings('ignore')

# Import hàm translate từ inference.py
from inference import translate

app = Flask(__name__)

# Load model khi khởi động app
print("Loading model...")
try:
    # Gọi translate một lần để load model vào cache
    translate("test")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route('/')
def home():
    """Trang chủ"""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate_text():
    """API endpoint để dịch văn bản"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Please enter some text to translate'
            })
        
        # Dịch văn bản
        translation = translate(text)
        
        return jsonify({
            'success': True,
            'source': text,
            'translation': translation
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
