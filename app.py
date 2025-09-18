import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

app = Flask(__name__)

print("🚀 ONC:ES - Batch Processing with Logo")

# Codificar logo a base64
def encode_logo(logo_path):
    """Convertir logo a base64 para HTML"""
    try:
        with open(logo_path, "rb") as logo_file:
            return base64.b64encode(logo_file.read()).decode('utf-8')
    except Exception as e:
        print(f"⚠️  Logo no encontrado: {e}")
        return None

LOGO_BASE64 = encode_logo('logo.png')

# Cargar modelo con nombre personalizado
def load_model():
    model_paths = ['m_final.keras', 'model.keras', 'model.h5']
    for path in model_paths:
        try:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path)
                print(f"✅ Modelo cargado desde: {path}")
                return model
        except Exception as e:
            print(f"❌ Error cargando {path}: {e}")
            continue
    
    print("⚠️  Creando modelo de demostración")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 60, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

model = load_model()

# HTML template con logo y múltiples imágenes
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ONC:ES - Batch Classification</title>
    <style>
        body {{ 
            font-family: 'Arial', sans-serif; 
            max-width: 800px; 
            margin: 50px auto; 
            padding: 20px;
            background: linear-gradient(135deg, #0c0c2d 0%, #1a1a4a 100%);
            color: white;
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 30px;
        }}
        .logo-img {{
            height: 100px;
            margin-bottom: 15px;
        }}
        .title {{
            font-size: 2.2em; 
            font-weight: bold; 
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 10px 0;
        }}
        .subtitle {{
            opacity: 0.8;
            margin-bottom: 20px;
        }}
        .upload-box {{ 
            border: 3px dashed #4ecdc4; 
            padding: 30px; 
            margin: 20px 0; 
            text-align: center;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.1);
        }}
        .file-input {{
            margin: 15px 0;
            padding: 12px;
            background: white;
            border-radius: 8px;
            color: #333;
            width: 80%;
            border: 2px solid transparent;
        }}
        .file-input:focus {{
            border-color: #4ecdc4;
            outline: none;
        }}
        .submit-btn {{
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 30px;
            font-size: 1.2em;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .submit-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        .info-box {{
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #ff6b6b;
        }}
        .feature-list {{
            text-align: left;
            margin: 15px 0;
        }}
        .feature-list li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        {logo_html}
        <div class="title">ONC:ES</div>
        <div class="subtitle">Optimized Neural Classificator: Elliptical vs Spiral</div>
        <p><em style="color: gray; font-style: italic;">Clasifícalo entonces</em>.</p>
    </div>
    
    <div class="info-box">
        <h3>🌌 Procesamiento de Lotes</h3>
        <ul class="feature-list">
            <li>✅ Sube múltiples imágenes simultáneamente</li>
            <li>✅ Procesamiento por lotes eficiente</li>
            <li>✅ Resultados en formato JSON/HTML</li>
        </ul>
    </div>
    
    <form action="/predict_batch" method="post" enctype="multipart/form-data">
        <div class="upload-box">
            <h3>📤 Subir Imágenes</h3>
            <input class="file-input" type="file" name="images" multiple accept=".jpg,.jpeg,.png,.fits">
            <br>
            <input class="submit-btn" type="submit" value="🚀 Clasificar Lote">
            <p><small>Selecciona múltiples archivos (Ctrl+Click)</small></p>
        </div>
    </form>
</body>
</html>
'''

def process_image(file):
    """Procesar una imagen individual"""
    try:
        img = Image.open(file.stream).convert('L')
        img = img.resize((60, 60))
        img_array = np.array(img) / 255.0
        img_array = np.stack([img_array, img_array], axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Error procesando imagen: {str(e)}")

def predict_image(img_array, filename):
    """Predecir una imagen y retornar resultados"""
    try:
        prediction = model.predict(img_array, verbose=0)
        prob = float(prediction[0][0])
        
        label = "Elíptica" if prob > 0.5 else "Espiral"
        confidence = prob if prob > 0.5 else 1 - prob
        
        return {
            'filename': filename,
            'label': label,
            'confidence': round(confidence, 4),
            'probability': round(prob, 4),
            'success': True
        }
    except Exception as e:
        raise Exception(f"Error en predicción: {str(e)}")

@app.route('/')
def home():
    logo_html = f'<img class="logo-img" src="data:image/png;base64,{LOGO_BASE64}" alt="ONC:ES Logo">' if LOGO_BASE64 else ''
    return HTML_TEMPLATE.format(logo_html=logo_html)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No se subieron imágenes'}), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No se seleccionaron archivos'}), 400

        print(f"📦 Procesando lote de {len(files)} imágenes...")
        start_time = datetime.now()
        
        results = []
        for file in files:
            if file.filename:
                try:
                    img_array = process_image(file)
                    result = predict_image(img_array, file.filename)
                    results.append(result)
                    print(f"✅ {file.filename}: {result['label']} ({result['confidence']:.2%})")
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'error': str(e),
                        'success': False
                    })
                    print(f"❌ Error en {file.filename}: {e}")

        # Estadísticas del lote
        processing_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r['success'])
        elliptical = sum(1 for r in results if r.get('label') == 'Elíptica')
        spiral = sum(1 for r in results if r.get('label') == 'Espiral')

        response_data = {
            'batch_id': start_time.strftime('%Y%m%d_%H%M%S'),
            'total_images': len(files),
            'successful_predictions': successful,
            'failed_predictions': len(files) - successful,
            'elliptical_count': elliptical,
            'spiral_count': spiral,
            'processing_time_seconds': round(processing_time, 2),
            'model': 'm_final.keras',
            'system': 'ONC:ES v1.0',
            'results': results,
            'success': True
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# Endpoint individual para compatibilidad
@app.route('/predict_single', methods=['POST'])
def predict_single():
    if 'image' not in request.files:
        return jsonify({'error': 'No se subió imagen'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo'}), 400

    try:
        img_array = process_image(file)
        result = predict_image(img_array, file.filename)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    print("🌌 ONC:ES System Initialized")
    print("📊 Model: m_final.keras")
    print("🚀 Server: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
