# convert_fits.py
from PIL import Image
import numpy as np
import os

def fits_to_jpeg_simple(fits_path, output_path):
    """Conversión simple de FITS a JPEG sin astropy"""
    try:
        # Leer como array numpy (simulación)
        # EN LA PRÁCTICA: Usa astropy cuando resuelvas la instalación
        # Por ahora creamos una imagen de prueba
        print(f"⚠️  Creando imagen de prueba para: {fits_path}")
        
        # Crear imagen de prueba
        img_array = np.random.rand(100, 100) * 255
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(output_path)
        print(f"✅ Imagen guardada: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Convertir tus archivos
fits_to_jpeg_simple('galaxy588007003633680516_i.fits', 'test_i.jpg')
fits_to_jpeg_simple('galaxy588007003633680516_u.fits', 'test_u.jpg')
