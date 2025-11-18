# ☕ Coffee Quality Classifier

## Descripción
Aplicación web completa que entrena un modelo de machine learning para clasificar la calidad del café, lo expone mediante una API REST con FastAPI y lo consume desde un formulario HTML moderno.

## 🎯 Objetivos del Proyecto
- **Data Scientist**: Entrena y limpia los datos
- **Engineer 1**: Crea main.py con FastAPI
- **Engineer 2**: Desarrolla formulario HTML
- **QA/Tester**: Realiza pruebas y validaciones de integración

## 🏗️ Arquitectura

### Componentes
1. **train_model.py** - Entrenamiento del modelo (Data Scientist)
2. **main.py** - API FastAPI (Engineer 1)
3. **static/index.html** - Interfaz web avanzada (Engineer 2)
4. **test_api.py** - Suite de pruebas (QA/Tester)

### Modelo de Machine Learning
- **Algoritmo**: Random Forest Classifier
- **Características**: Acidez, Dulzura, Cuerpo, Aroma, Altitud
- **Clases**: Premium, Bueno, Regular
- **Preprocesamiento**: StandardScaler

## 🚀 Instalación y Ejecución

### 1. Instalar dependencias
```bash
pip install fastapi uvicorn scikit-learn pandas numpy python-multipart requests
```

O usando requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo (Data Scientist)
```bash
python train_model.py
```
Esto generará el archivo `model.pkl` necesario para la API.

### 3. Ejecutar la API (Engineer 1)
```bash
python main.py
```
O alternativamente:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Acceder a la aplicación
- **Interfaz web**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 5. Ejecutar pruebas (QA/Tester)
```bash
python test_api.py
```

## 📊 Características del Modelo

### Entradas
- **Acidez** (1-10): Nivel de acidez perceptible
- **Dulzura** (1-10): Intensidad del sabor dulce natural
- **Cuerpo** (1-10): Peso y textura en boca
- **Aroma** (1-10): Intensidad y calidad del aroma
- **Altitud** (500-2000m): Elevación del cultivo

### Salidas
- **Premium**: Café de alta calidad
- **Bueno**: Café de calidad media-alta
- **Regular**: Café de calidad básica

## 🔌 API Endpoints

### GET /
- **Descripción**: Página principal con formulario web
- **Respuesta**: HTML con interfaz completa

### POST /predict
- **Descripción**: Predicción usando datos de formulario
- **Entrada**: Form data (acidity, sweetness, body, aroma, altitude)
- **Respuesta**: JSON con quality, confidence, features

### POST /predict-json
- **Descripción**: Predicción usando JSON
- **Entrada**: JSON con características del café
- **Respuesta**: JSON con predicción y confianza

### GET /health
- **Descripción**: Estado de la API y modelo
- **Respuesta**: Status, model_loaded, accuracy

### GET /model-info
- **Descripción**: Información detallada del modelo
- **Respuesta**: Features, accuracy, classes

## 🧪 Testing (QA/Tester)

### Pruebas Incluidas
1. **Health Check**: Verificar estado de la API
2. **Model Info**: Información del modelo
3. **Main Page**: Carga de página principal
4. **Form Prediction**: Predicción via formulario
5. **JSON Prediction**: Predicción via JSON API
6. **Input Validation**: Validación de entradas
7. **Response Time**: Tiempo de respuesta
8. **Concurrent Requests**: Peticiones concurrentes

### Ejecutar Pruebas
```bash
# Pruebas locales
python test_api.py

# Pruebas en URL específica
python test_api.py http://tu-url-replit.com
```

## 📱 Interfaces de Usuario

### Interfaz Principal (/)
- Formulario interactivo con validación
- Ejemplos predefinidos
- Diseño responsive y moderno
- Indicadores de carga y resultados animados

### Interfaz Avanzada (/static/index.html)
- Diseño más elaborado con gradientes
- Tarjetas de ejemplos clicables
- Validación en tiempo real
- Animaciones y efectos visuales

## 🔧 Para Replit

### 1. Archivos necesarios
Asegúrate de tener todos estos archivos en tu Replit:
- `main.py`
- `train_model.py`
- `test_api.py`
- `requirements.txt`
- `model.pkl` (generado después de ejecutar train_model.py)
- `static/index.html`

### 2. Configuración
1. Ejecuta primero: `python train_model.py`
2. Luego haz clic en "Run" (ejecutará main.py)
3. La URL aparecerá en el panel derecho

### 3. Testing en Replit
```bash
python test_api.py https://tu-replit-url.com
```

## 👥 Roles del Equipo

### 📊 Data Scientist
- **Archivo**: `train_model.py`
- **Responsabilidades**:
  - Generación de dataset sintético
  - Limpieza y preprocesamiento
  - Entrenamiento del modelo Random Forest
  - Evaluación y métricas
  - Persistencia del modelo

### ⚙️ Engineer 1 (Backend)
- **Archivo**: `main.py`
- **Responsabilidades**:
  - API REST con FastAPI
  - Endpoints de predicción
  - Validación de datos
  - Manejo de errores
  - Documentación automática

### 🎨 Engineer 2 (Frontend)
- **Archivo**: `static/index.html`
- **Responsabilidades**:
  - Interfaz web responsive
  - Formularios interactivos
  - Validación client-side
  - UX/UI moderno
  - Integración con API

### 🧪 QA/Tester
- **Archivo**: `test_api.py`
- **Responsabilidades**:
  - Suite de pruebas automatizadas
  - Validación de integración
  - Tests de rendimiento
  - Verificación de endpoints
  - Reporting de resultados

## 📈 Ejemplos de Uso

### Café Premium
```json
{
    "acidity": 5.5,
    "sweetness": 8.0,
    "body": 7.5,
    "aroma": 8.5,
    "altitude": 1500
}
```

### Café Bueno
```json
{
    "acidity": 6.0,
    "sweetness": 6.5,
    "body": 6.0,
    "aroma": 6.8,
    "altitude": 1200
}
```

### Café Regular
```json
{
    "acidity": 4.0,
    "sweetness": 4.5,
    "body": 5.0,
    "aroma": 5.2,
    "altitude": 800
}
```

## 🔍 Troubleshooting

### Problema: Modelo no carga
**Solución**: Ejecuta `python train_model.py` para generar `model.pkl`

### Problema: Error 503 en predicción
**Solución**: Verifica que `model.pkl` esté en el directorio raíz

### Problema: Tests fallan
**Solución**: Asegúrate de que la API esté ejecutándose en el puerto correcto

### Problema: Página no carga en Replit
**Solución**: Verifica que el puerto sea 8000 y esté configurado correctamente

## 📝 Notas Adicionales

- El modelo usa datos sintéticos pero sigue patrones realistas de calidad de café
- La API incluye documentación automática en `/docs`
- Los tests son ejecutables tanto local como remotamente
- El diseño es completamente responsive para móviles

## 🏆 Entregables

- ✅ Modelo entrenado y persistido
- ✅ API REST funcional con FastAPI
- ✅ Interfaz web moderna y responsive
- ✅ Suite completa de pruebas
- ✅ Documentación completa
- ✅ Link de Replit público del grupo
# coffee-quality-classifier-ml-api
