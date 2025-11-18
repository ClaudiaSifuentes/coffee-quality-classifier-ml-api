"""
Data Scientist - Entrenamiento y limpieza de datos
Modelo de clasificación de café basado en características
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_coffee_dataset():
    """
    Crear un dataset sintético de café con diferentes características
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Características del café
    acidity = np.random.normal(5.0, 1.5, n_samples)  # pH 3-7
    sweetness = np.random.normal(6.0, 2.0, n_samples)  # Escala 1-10
    body = np.random.normal(7.0, 1.8, n_samples)  # Escala 1-10
    aroma = np.random.normal(6.5, 1.5, n_samples)  # Escala 1-10
    altitude = np.random.normal(1200, 300, n_samples)  # metros sobre el nivel del mar
    
    # Crear etiquetas basadas en reglas lógicas
    quality_labels = []
    for i in range(n_samples):
        score = 0
        
        # Reglas para determinar calidad
        if 4.5 <= acidity[i] <= 6.0:
            score += 1
        if sweetness[i] >= 6.0:
            score += 1
        if body[i] >= 6.5:
            score += 1
        if aroma[i] >= 6.0:
            score += 1
        if altitude[i] >= 1000:
            score += 1
            
        # Clasificación basada en score
        if score >= 4:
            quality_labels.append('Premium')
        elif score >= 2:
            quality_labels.append('Bueno')
        else:
            quality_labels.append('Regular')
    
    # Crear DataFrame
    data = pd.DataFrame({
        'acidity': acidity,
        'sweetness': sweetness,
        'body': body,
        'aroma': aroma,
        'altitude': altitude,
        'quality': quality_labels
    })
    
    # Limpiar datos (eliminar outliers)
    data = data[
        (data['acidity'] >= 1) & (data['acidity'] <= 10) &
        (data['sweetness'] >= 1) & (data['sweetness'] <= 10) &
        (data['body'] >= 1) & (data['body'] <= 10) &
        (data['aroma'] >= 1) & (data['aroma'] <= 10) &
        (data['altitude'] >= 500) & (data['altitude'] <= 2000)
    ]
    
    return data

def train_model():
    """
    Entrenar el modelo de clasificación
    """
    print("📊 Generando dataset de café...")
    data = create_coffee_dataset()
    
    print(f"📈 Dataset creado con {len(data)} muestras")
    print("Distribución de calidades:")
    print(data['quality'].value_counts())
    
    # Preparar características y etiquetas
    X = data[['acidity', 'sweetness', 'body', 'aroma', 'altitude']]
    y = data['quality']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    print("🤖 Entrenando modelo Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluación
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Accuracy del modelo: {accuracy:.3f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo y scaler
    print("💾 Guardando modelo...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'accuracy': accuracy
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("✅ Modelo guardado como 'model.pkl'")
    
    return model_data

if __name__ == "__main__":
    train_model()
