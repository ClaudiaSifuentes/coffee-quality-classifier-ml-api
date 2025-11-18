"""
Engineer 1 - API principal con FastAPI
Exponer modelo de clasificación de café
"""

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from typing import Dict, Any
import os

# Crear aplicación FastAPI
app = FastAPI(
    title="Coffee Quality Classifier API",
    description="API para clasificar la calidad del café basado en sus características",
    version="1.0.0"
)

# Modelo global
model_data = None

class CoffeeFeatures(BaseModel):
    """Modelo Pydantic para las características del café"""
    acidity: float
    sweetness: float
    body: float
    aroma: float
    altitude: float
    
    class Config:
        schema_extra = {
            "example": {
                "acidity": 5.5,
                "sweetness": 7.0,
                "body": 6.8,
                "aroma": 7.2,
                "altitude": 1200
            }
        }

class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    quality: str
    confidence: float
    features: Dict[str, float]

def load_model():
    """Cargar el modelo entrenado"""
    global model_data
    try:
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            print("✅ Modelo cargado exitosamente")
            print(f"📊 Accuracy del modelo: {model_data['accuracy']:.3f}")
        else:
            print("❌ Archivo model.pkl no encontrado. Ejecuta train_model.py primero.")
            model_data = None
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        model_data = None

# Cargar modelo al iniciar
load_model()

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    print("🚀 Coffee Quality Classifier API iniciada")
    if model_data is None:
        print("⚠️ Modelo no cargado. Algunas funcionalidades no estarán disponibles.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Página principal con formulario"""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clasificador de Calidad de Café</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #8B4513 0%, #D2691E 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            }
            h1 {
                color: #8B4513;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #5D4037;
            }
            input[type="number"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #D7CCC8;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input[type="number"]:focus {
                border-color: #8B4513;
                outline: none;
            }
            .btn {
                background: #8B4513;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                cursor: pointer;
                width: 100%;
                transition: background-color 0.3s;
            }
            .btn:hover {
                background: #6D4C41;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 8px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .result.premium {
                background: #E8F5E8;
                border: 2px solid #4CAF50;
                color: #2E7D32;
            }
            .result.bueno {
                background: #FFF3E0;
                border: 2px solid #FF9800;
                color: #E65100;
            }
            .result.regular {
                background: #FFEBEE;
                border: 2px solid #F44336;
                color: #C62828;
            }
            .info {
                background: #E3F2FD;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #2196F3;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
                padding: 20px;
                background: #f0f0f0;
                border-radius: 8px;
                border: 2px solid #ddd;
            }
            .loading.show {
                display: block !important;
            }
            .coffee-icon {
                font-size: 3em;
                text-align: center;
                margin-bottom: 20px;
            }
            /* Asegurar que los resultados sean visibles */
            .result {
                margin-top: 20px !important;
                padding: 20px !important;
                border-radius: 8px !important;
                display: none !important;
            }
            .result.show {
                display: block !important;
                animation: fadeIn 0.5s ease-in;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="coffee-icon">☕</div>
            <h1>Clasificador de Calidad de Café</h1>
            
            <div class="info">
                <strong>ℹ️ Instrucciones:</strong><br>
                Ingresa las características de tu café para obtener una clasificación de calidad.
                <ul>
                    <li><strong>Acidez:</strong> Nivel de acidez (1-10)</li>
                    <li><strong>Dulzura:</strong> Nivel de dulzura (1-10)</li>
                    <li><strong>Cuerpo:</strong> Intensidad del cuerpo (1-10)</li>
                    <li><strong>Aroma:</strong> Intensidad del aroma (1-10)</li>
                    <li><strong>Altitud:</strong> Metros sobre el nivel del mar (500-2000)</li>
                </ul>
            </div>

            <form id="coffeeForm">
                <div class="form-group">
                    <label for="acidity">Acidez (1-10):</label>
                    <input type="number" id="acidity" name="acidity" min="1" max="10" step="0.1" value="5.5" required>
                </div>
                
                <div class="form-group">
                    <label for="sweetness">Dulzura (1-10):</label>
                    <input type="number" id="sweetness" name="sweetness" min="1" max="10" step="0.1" value="7.0" required>
                </div>
                
                <div class="form-group">
                    <label for="body">Cuerpo (1-10):</label>
                    <input type="number" id="body" name="body" min="1" max="10" step="0.1" value="6.8" required>
                </div>
                
                <div class="form-group">
                    <label for="aroma">Aroma (1-10):</label>
                    <input type="number" id="aroma" name="aroma" min="1" max="10" step="0.1" value="7.2" required>
                </div>
                
                <div class="form-group">
                    <label for="altitude">Altitud (metros):</label>
                    <input type="number" id="altitude" name="altitude" min="500" max="2000" step="1" value="1200" required>
                </div>
                
                <button type="submit" class="btn">🔍 Clasificar Café</button>
            </form>
            
            <div class="loading" id="loading">
                <p>⏳ Analizando café...</p>
            </div>
            
            <div class="result" id="result">
                <h3 id="resultTitle"></h3>
                <p id="resultText"></p>
                <p id="confidenceText"></p>
            </div>
        </div>

        <script>
            document.getElementById('coffeeForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                console.log('Formulario enviado'); // Debug
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                // Mostrar loading
                loading.className = 'loading show';
                result.className = 'result';
                
                // Obtener valores del formulario
                const acidity = document.getElementById('acidity').value;
                const sweetness = document.getElementById('sweetness').value;
                const body = document.getElementById('body').value;
                const aroma = document.getElementById('aroma').value;
                const altitude = document.getElementById('altitude').value;
                
                console.log('Valores:', { acidity, sweetness, body, aroma, altitude }); // Debug
                
                const formData = new FormData();
                formData.append('acidity', acidity);
                formData.append('sweetness', sweetness);
                formData.append('body', body);
                formData.append('aroma', aroma);
                formData.append('altitude', altitude);
                
                try {
                    console.log('Enviando petición...'); // Debug
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    console.log('Respuesta recibida:', response.status); // Debug
                    const data = await response.json();
                    console.log('Datos:', data); // Debug
                    
                    if (response.ok) {
                        const resultTitle = document.getElementById('resultTitle');
                        const resultText = document.getElementById('resultText');
                        const confidenceText = document.getElementById('confidenceText');
                        
                        resultTitle.textContent = `Calidad: ${data.quality}`;
                        resultText.textContent = getQualityDescription(data.quality);
                        confidenceText.textContent = `Confianza: ${(data.confidence * 100).toFixed(1)}%`;
                        
                        result.className = `result show ${data.quality.toLowerCase()}`;
                        console.log('Resultado mostrado correctamente'); // Debug
                    } else {
                        throw new Error(data.detail || 'Error desconocido');
                    }
                } catch (error) {
                    console.error('Error:', error); // Debug
                    const resultTitle = document.getElementById('resultTitle');
                    const resultText = document.getElementById('resultText');
                    const confidenceText = document.getElementById('confidenceText');
                    
                    resultTitle.textContent = 'Error';
                    resultText.textContent = `Error: ${error.message}`;
                    confidenceText.textContent = '';
                    result.className = 'result show regular';
                }
                
                loading.className = 'loading';
                console.log('Loading oculto'); // Debug
            });
            
            function getQualityDescription(quality) {
                const descriptions = {
                    'Premium': '🏆 ¡Excelente café! Este café tiene características superiores que lo hacen ideal para los paladares más exigentes.',
                    'Bueno': '👍 Buen café con características sólidas. Una elección confiable para el consumo diario.',
                    'Regular': '⚠️ Café de calidad básica. Podría beneficiarse de mejoras en el procesamiento o origen.'
                };
                return descriptions[quality] || 'Calidad no reconocida.';
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict", response_model=PredictionResponse)
async def predict_coffee_quality(
    acidity: float = Form(...),
    sweetness: float = Form(...),
    body: float = Form(...),
    aroma: float = Form(...),
    altitude: float = Form(...)
):
    """Predecir la calidad del café basado en características"""
    
    if model_data is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Entrena el modelo primero.")
    
    try:
        # Validar entrada
        features = {
            'acidity': acidity,
            'sweetness': sweetness,
            'body': body,
            'aroma': aroma,
            'altitude': altitude
        }
        
        # Validaciones básicas
        if not (1 <= acidity <= 10):
            raise HTTPException(status_code=400, detail="Acidez debe estar entre 1 y 10")
        if not (1 <= sweetness <= 10):
            raise HTTPException(status_code=400, detail="Dulzura debe estar entre 1 y 10")
        if not (1 <= body <= 10):
            raise HTTPException(status_code=400, detail="Cuerpo debe estar entre 1 y 10")
        if not (1 <= aroma <= 10):
            raise HTTPException(status_code=400, detail="Aroma debe estar entre 1 y 10")
        if not (500 <= altitude <= 2000):
            raise HTTPException(status_code=400, detail="Altitud debe estar entre 500 y 2000 metros")
        
        # Preparar datos para predicción
        feature_array = np.array([[acidity, sweetness, body, aroma, altitude]])
        feature_array_scaled = model_data['scaler'].transform(feature_array)
        
        # Hacer predicción
        prediction = model_data['model'].predict(feature_array_scaled)[0]
        prediction_proba = model_data['model'].predict_proba(feature_array_scaled)[0]
        
        # Obtener confianza
        confidence = float(max(prediction_proba))
        
        return PredictionResponse(
            quality=prediction,
            confidence=confidence,
            features=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict-json", response_model=PredictionResponse)
async def predict_coffee_quality_json(features: CoffeeFeatures):
    """Predecir calidad del café usando JSON (para APIs)"""
    
    if model_data is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Entrena el modelo primero.")
    
    try:
        # Preparar datos
        feature_array = np.array([[
            features.acidity, features.sweetness, features.body, 
            features.aroma, features.altitude
        ]])
        feature_array_scaled = model_data['scaler'].transform(feature_array)
        
        # Predicción
        prediction = model_data['model'].predict(feature_array_scaled)[0]
        prediction_proba = model_data['model'].predict_proba(feature_array_scaled)[0]
        confidence = float(max(prediction_proba))
        
        return PredictionResponse(
            quality=prediction,
            confidence=confidence,
            features=features.dict()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
async def health_check():
    """Verificar estado de la API"""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "model_accuracy": model_data['accuracy'] if model_data else None
    }

@app.get("/model-info")
async def model_info():
    """Información del modelo"""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return {
        "features": model_data['feature_names'],
        "accuracy": model_data['accuracy'],
        "classes": list(model_data['model'].classes_)
    }

if __name__ == "__main__":
    print("🚀 Iniciando Coffee Quality Classifier API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
