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
import re
from astropy.io import fits

app = Flask(__name__)

print("🚀 ONC:ES - Dual Band Processing")

# Codificar logo
def encode_logo(logo_path):
    try:
        with open(logo_path, "rb") as logo_file:
            return base64.b64encode(logo_file.read()).decode('utf-8')
    except:
        return None

LOGO_BASE64 = encode_logo('logo.png')

# Cargar modelo
def load_model():
    path = 'm_final.keras'
    model = tf.keras.models.load_model(path)
    print("✅ Modelo cargado")
    return model


model = load_model()

def extract_galaxy_id(filename):
    """Extraer ID de galaxia del nombre de archivo"""
    # Ejemplo: "galaxy123_g.jpg" → "galaxy123"
    patterns = [
        r'(.+?)[_\-](u|g|r|i|u|z)\.',  # galaxy123_g.jpg
        r'(.+?)(u|g|r|i|u|z)\.',       # galaxy123g.jpg
        r'(.+?)\.'                    # galaxy123.jpg
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return filename  # Si no se encuentra patrón, usar nombre completo

# HTML template para dos bandas
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ONC:ES - Optimized Neural Classifier: Elliptical vs. Spiral</title>
    <style>
        body {{ 
            font-family: 'Arial', sans-serif; 
            max-width: 900px; 
            margin: 50px auto; 
            padding: 20px;
            background: linear-gradient(135deg, #0c0c2d 0%, #1a1a4a 100%);
            color: white;
        }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .logo-img {{ height: 100px; margin-bottom: 15px; }}
        .title {{ 
            font-size: 2.2em; 
            font-weight: bold; 
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .band-section {{
            display: inline-block;
            width: 45%;
            margin: 10px;
            vertical-align: top;
        }}
        .upload-box {{ 
            border: 3px dashed #4ecdc4; 
            padding: 20px; 
            margin: 15px 0; 
            text-align: center;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.1);
        }}
        .band-g {{ border-color: #4ecdc4; }}   /* Turquesa para banda g */
        .band-r {{ border-color: #ff6b6b; }}   /* Coral para banda r */
        .file-input {{
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
            color: #333;
            width: 90%;
        }}
        .submit-btn {{
            background: linear-gradient(45deg, #4ecdc4, #ff6b6b);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            margin: 20px 0;
        }}
        .instructions {{
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        {logo_html}
        <div class="title">ONC:ES - Optimized Neural Classifier</div>
        <div>Banda g + Banda r → Clasificación</div>
    </div>
    
    <div class="instructions">
        <h3>📋 Instrucciones:</h3>
        <ol>
            <li>Sube imágenes de <strong>banda g</strong> en el primer buzón</li>
            <li>Sube imágenes de <strong>banda r</strong> en el segundo buzón</li>
            <li>Mismo número de archivos en ambos buzones</li>
            <li>Mismo orden de galaxias en ambos conjuntos</li>
            <li>Click en "Clasificar Galaxias"</li>
        </ol>
    </div>
    
    <form action="/predict_dual_band" method="post" enctype="multipart/form-data">
        <div class="band-section">
            <div class="upload-box band-g">
                <h3>🌌 Banda g</h3>
                <input class="file-input" type="file" name="band_g" multiple accept=".jpg,.jpeg,.png">
                <p>Imágenes en banda g</p>
            </div>
        </div>
        
        <div class="band-section">
            <div class="upload-box band-i">
                <h3>🌌 Banda r</h3>
                <input class="file-input" type="file" name="band_r" multiple accept=".jpg,.jpeg,.png">
                <p>Imágenes en banda r</p>
            </div>
        </div>
        
        <div style="text-align: center; clear: both;">
            <input class="submit-btn" type="submit" value="🔍 Clasificar Galaxias">
        </div>
    </form>
</body>
</html>
'''

@app.route('/')
def home():
    logo_html = f'<img class="logo-img" src="data:image/png;base64,{LOGO_BASE64}" alt="ONC:ES Logo">' if LOGO_BASE64 else ''
    return HTML_TEMPLATE.format(logo_html=logo_html)

@app.route('/predict_dual_band', methods=['POST'])
def predict_dual_band():
    try:
        # Verificar que se subieron ambas bandas (CORREGIDO: band_r en lugar de band_i)
        if 'band_g' not in request.files or 'band_r' not in request.files:
            return jsonify({'error': 'Se requieren ambas bandas (g y r)'}), 400
        
        band_g_files = request.files.getlist('band_g')
        band_r_files = request.files.getlist('band_r')  # Corregido: band_r en lugar de band_i
        
        # Verificación corregida
        if not band_g_files or not band_r_files or band_g_files[0].filename == '' or band_r_files[0].filename == '':
            return jsonify({'error': 'Ambas bandas deben contener archivos'}), 400
        
        # Verificación de cantidad corregida
        if len(band_g_files) != len(band_r_files):
            return jsonify({
                'error': f'Diferente número de archivos: {len(band_g_files)} en banda g vs {len(band_r_files)} en banda r'
            }), 400

        print(f"📦 Procesando {len(band_g_files)} pares de imágenes...")
        start_time = datetime.now()
        
        imageList = []
        galaxy_id_list = []
        successful_pairs = 0
        failed_pairs = 0
        
        # Procesar cada par de imágenes
        for i, (g_file, r_file) in enumerate(zip(band_g_files, band_r_files)):
            try:
                galaxy_id = extract_galaxy_id(g_file.filename)
                galaxy_id_list.append(galaxy_id)
                
                # Leer y procesar archivo FITS de banda g
                g_file.save(f'temp_g_{i}.fits')
                with fits.open(f'temp_g_{i}.fits') as hdul:
                    g_data = np.nan_to_num(hdul[0].data)
                os.remove(f'temp_g_{i}.fits')
                
                # Leer y procesar archivo FITS de banda r
                r_file.save(f'temp_r_{i}.fits')
                with fits.open(f'temp_r_{i}.fits') as hdul:
                    r_data = np.nan_to_num(hdul[0].data)
                os.remove(f'temp_r_{i}.fits')
                
                # Redimensionar y concatenar
                g_resized = tf.image.resize(np.expand_dims(g_data, axis=-1), (60, 60))
                r_resized = tf.image.resize(np.expand_dims(r_data, axis=-1), (60, 60))
                
                img = np.concatenate([g_resized, r_resized], axis=-1)
                imageList.append(img)
                
                successful_pairs += 1
                print(f"✅ Par {i+1}: {galaxy_id} procesado correctamente")
                
            except Exception as e:
                failed_pairs += 1
                print(f"❌ Error en par {i+1}: {e}")

        # Realizar predicción
        x = np.stack(imageList)
        predict = model.predict(x, verbose=0)
        y_pred = (predict > .5479).astype(int)
        
        # Preparar resultados
        results = []
        for i, pred in enumerate(y_pred):
            label = 'Elíptica' if pred[0] == 1 else 'Espiral'
            results.append({
                "galaxy_id": galaxy_id_list[i],
                "label": label,
                "confidence": float(predict[i][0])
            })

        # Estadísticas
        processing_time = (datetime.now() - start_time).total_seconds()
        elliptical = sum(1 for r in results if r.get('label') == 'Elíptica')
        spiral = sum(1 for r in results if r.get('label') == 'Espiral')

        return jsonify({
            'batch_id': start_time.strftime('%Y%m%d_%H%M%S'),
            'total_pairs': len(band_g_files),
            'successful_pairs': successful_pairs,
            'failed_pairs': failed_pairs,
            'elliptical_count': elliptical,
            'spiral_count': spiral,
            'processing_time_seconds': round(processing_time, 2),
            'model': 'm_final.keras',
            'system': 'ONC:ES v1.0',
            'results': results,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': 'ONC:ES v1.0 - Dual Band'
    })

if __name__ == '__main__':
    print("🌌 ONC:ES Optimized Neural Classifier: Elliptical vs Spiral")
    print("📊 Model: m_final.keras")
    print("🎯 Endpoint: /predict_dual_band")
    print("🚀 Server: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
