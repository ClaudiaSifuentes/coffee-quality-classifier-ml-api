"""
QA/Tester - Pruebas y validaciones de integración
Script completo de testing para la aplicación de clasificación de café
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

class CoffeeAPITester:
    """Clase para realizar pruebas completas de la API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Registrar resultado de test"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        result = f"{status} - {test_name}"
        if message:
            result += f": {message}"
            
        print(result)
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        
    def test_api_health(self):
        """Test 1: Verificar estado de la API"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_test("API Health Check", True, f"API saludable, modelo cargado: {data.get('model_loaded')}")
                else:
                    self.log_test("API Health Check", False, "API no saludable")
            else:
                self.log_test("API Health Check", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("API Health Check", False, str(e))
    
    def test_model_info(self):
        """Test 2: Información del modelo"""
        try:
            response = requests.get(f"{self.base_url}/model-info", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['features', 'accuracy', 'classes']
                
                if all(field in data for field in required_fields):
                    accuracy = data.get('accuracy', 0)
                    self.log_test("Model Info", True, f"Accuracy: {accuracy:.3f}, Classes: {len(data.get('classes', []))}")
                else:
                    self.log_test("Model Info", False, "Campos faltantes en respuesta")
            else:
                self.log_test("Model Info", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Model Info", False, str(e))
    
    def test_main_page(self):
        """Test 3: Página principal"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            
            if response.status_code == 200:
                content = response.text
                if ("Clasificador de Calidad de Café" in content and 
                    "form" in content.lower() and 
                    "acidity" in content):
                    self.log_test("Main Page", True, "Página principal cargada correctamente")
                else:
                    self.log_test("Main Page", False, "Contenido incompleto en página principal")
            else:
                self.log_test("Main Page", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Main Page", False, str(e))
    
    def test_prediction_form(self):
        """Test 4: Predicción mediante formulario"""
        test_cases = [
            {
                "name": "Café Premium",
                "data": {
                    "acidity": 5.5,
                    "sweetness": 8.0,
                    "body": 7.5,
                    "aroma": 8.5,
                    "altitude": 1500
                },
                "expected_quality": "Premium"
            },
            {
                "name": "Café Bueno",
                "data": {
                    "acidity": 6.0,
                    "sweetness": 6.5,
                    "body": 6.0,
                    "aroma": 6.8,
                    "altitude": 1200
                },
                "expected_quality": "Bueno"
            },
            {
                "name": "Café Regular",
                "data": {
                    "acidity": 4.0,
                    "sweetness": 4.5,
                    "body": 5.0,
                    "aroma": 5.2,
                    "altitude": 800
                },
                "expected_quality": "Regular"
            }
        ]
        
        for case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    data=case["data"],
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_quality = result.get('quality')
                    confidence = result.get('confidence', 0)
                    
                    if predicted_quality == case["expected_quality"]:
                        self.log_test(
                            f"Predicción Form - {case['name']}", 
                            True, 
                            f"Predicho: {predicted_quality}, Confianza: {confidence:.3f}"
                        )
                    else:
                        self.log_test(
                            f"Predicción Form - {case['name']}", 
                            False, 
                            f"Esperado: {case['expected_quality']}, Obtenido: {predicted_quality}"
                        )
                else:
                    self.log_test(
                        f"Predicción Form - {case['name']}", 
                        False, 
                        f"Status code: {response.status_code}"
                    )
                    
            except Exception as e:
                self.log_test(f"Predicción Form - {case['name']}", False, str(e))
    
    def test_prediction_json(self):
        """Test 5: Predicción mediante JSON"""
        test_data = {
            "acidity": 5.5,
            "sweetness": 7.0,
            "body": 6.8,
            "aroma": 7.2,
            "altitude": 1200
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/predict-json",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                required_fields = ['quality', 'confidence', 'features']
                
                if all(field in result for field in required_fields):
                    quality = result.get('quality')
                    confidence = result.get('confidence', 0)
                    self.log_test(
                        "Predicción JSON", 
                        True, 
                        f"Calidad: {quality}, Confianza: {confidence:.3f}"
                    )
                else:
                    self.log_test("Predicción JSON", False, "Campos faltantes en respuesta")
            else:
                self.log_test("Predicción JSON", False, f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_test("Predicción JSON", False, str(e))
    
    def test_input_validation(self):
        """Test 6: Validación de entradas"""
        invalid_cases = [
            {
                "name": "Acidez fuera de rango",
                "data": {"acidity": 15, "sweetness": 7, "body": 6, "aroma": 7, "altitude": 1200}
            },
            {
                "name": "Altitud fuera de rango",
                "data": {"acidity": 5, "sweetness": 7, "body": 6, "aroma": 7, "altitude": 3000}
            },
            {
                "name": "Valores negativos",
                "data": {"acidity": -1, "sweetness": 7, "body": 6, "aroma": 7, "altitude": 1200}
            }
        ]
        
        for case in invalid_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    data=case["data"],
                    timeout=5
                )
                
                if response.status_code == 400:
                    self.log_test(f"Validación - {case['name']}", True, "Error 400 capturado correctamente")
                elif response.status_code == 422:
                    self.log_test(f"Validación - {case['name']}", True, "Error 422 capturado correctamente")
                else:
                    self.log_test(
                        f"Validación - {case['name']}", 
                        False, 
                        f"Se esperaba error 400/422, obtenido: {response.status_code}"
                    )
                    
            except Exception as e:
                self.log_test(f"Validación - {case['name']}", False, str(e))
    
    def test_response_time(self):
        """Test 7: Tiempo de respuesta"""
        test_data = {
            "acidity": 5.5,
            "sweetness": 7.0,
            "body": 6.8,
            "aroma": 7.2,
            "altitude": 1200
        }
        
        times = []
        for i in range(5):
            try:
                start_time = time.time()
                response = requests.post(f"{self.base_url}/predict", data=test_data, timeout=10)
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
                    
            except Exception as e:
                self.log_test("Tiempo de Respuesta", False, f"Error en iteración {i+1}: {str(e)}")
                return
        
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            if avg_time < 2.0:  # Menos de 2 segundos promedio
                self.log_test(
                    "Tiempo de Respuesta", 
                    True, 
                    f"Promedio: {avg_time:.3f}s, Máximo: {max_time:.3f}s"
                )
            else:
                self.log_test(
                    "Tiempo de Respuesta", 
                    False, 
                    f"Tiempo promedio muy alto: {avg_time:.3f}s"
                )
    
    def test_concurrent_requests(self):
        """Test 8: Peticiones concurrentes"""
        import threading
        
        test_data = {
            "acidity": 5.5,
            "sweetness": 7.0,    
            "body": 6.8,
            "aroma": 7.2,
            "altitude": 1200
        }
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = requests.post(f"{self.base_url}/predict", data=test_data, timeout=10)
                if response.status_code == 200:
                    results.append(response.json())
                else:
                    errors.append(f"Status: {response.status_code}")
            except Exception as e:
                errors.append(str(e))
        
        # Crear 10 threads concurrentes
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen todos
        for thread in threads:
            thread.join()
        
        success_rate = len(results) / (len(results) + len(errors)) * 100
        
        if success_rate >= 80:  # Al menos 80% de éxito
            self.log_test(
                "Peticiones Concurrentes", 
                True, 
                f"Éxito: {len(results)}/10 ({success_rate:.1f}%)"
            )
        else:
            self.log_test(
                "Peticiones Concurrentes", 
                False, 
                f"Tasa de éxito baja: {success_rate:.1f}%"
            )
    
    def run_all_tests(self):
        """Ejecutar todas las pruebas"""
        print("🧪 Iniciando batería completa de pruebas...\n")
        print("=" * 60)
        
        # Verificar si la API está disponible
        try:
            requests.get(f"{self.base_url}/health", timeout=5)
        except:
            print("❌ No se puede conectar a la API. Asegúrate de que esté ejecutándose en", self.base_url)
            return
        
        # Ejecutar tests
        print("🔍 Ejecutando pruebas...")
        print("-" * 40)
        
        self.test_api_health()
        self.test_model_info()
        self.test_main_page()
        self.test_prediction_form()
        self.test_prediction_json()
        self.test_input_validation()
        self.test_response_time()
        self.test_concurrent_requests()
        
        # Resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE PRUEBAS")
        print("=" * 60)
        print(f"Total de pruebas: {self.total_tests}")
        print(f"Pruebas exitosas: {self.passed_tests}")
        print(f"Pruebas fallidas: {self.total_tests - self.passed_tests}")
        print(f"Tasa de éxito: {(self.passed_tests/self.total_tests*100):.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
            return True
        else:
            print(f"\n⚠️ {self.total_tests - self.passed_tests} pruebas fallaron.")
            print("\nPruebas fallidas:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['message']}")
            return False

def integration_test():
    """Prueba de integración completa"""
    print("🔄 PRUEBA DE INTEGRACIÓN COMPLETA")
    print("=" * 50)
    
    # Datos de prueba para flujo completo
    test_scenarios = [
        {
            "name": "Escenario 1: Café de Alta Calidad",
            "data": {"acidity": 5.8, "sweetness": 8.2, "body": 7.8, "aroma": 8.0, "altitude": 1600}
        },
        {
            "name": "Escenario 2: Café Comercial",
            "data": {"acidity": 6.2, "sweetness": 6.0, "body": 6.5, "aroma": 6.3, "altitude": 1100}
        },
        {
            "name": "Escenario 3: Café Básico",
            "data": {"acidity": 4.5, "sweetness": 5.0, "body": 5.5, "aroma": 5.0, "altitude": 900}
        }
    ]
    
    base_url = "http://localhost:8000"
    
    for scenario in test_scenarios:
        print(f"\n🧪 {scenario['name']}")
        print("-" * 30)
        
        try:
            # Test formulario
            response_form = requests.post(f"{base_url}/predict", data=scenario['data'])
            
            # Test JSON
            response_json = requests.post(
                f"{base_url}/predict-json", 
                json=scenario['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response_form.status_code == 200 and response_json.status_code == 200:
                result_form = response_form.json()
                result_json = response_json.json()
                
                print(f"✅ Formulario: {result_form['quality']} ({result_form['confidence']:.3f})")
                print(f"✅ JSON API: {result_json['quality']} ({result_json['confidence']:.3f})")
                
                # Verificar consistencia
                if result_form['quality'] == result_json['quality']:
                    print("✅ Consistencia: Ambos métodos devuelven el mismo resultado")
                else:
                    print("⚠️ Inconsistencia: Resultados diferentes entre métodos")
            else:
                print(f"❌ Error: Form={response_form.status_code}, JSON={response_json.status_code}")
                
        except Exception as e:
            print(f"❌ Error en {scenario['name']}: {str(e)}")

if __name__ == "__main__":
    print("🚀 COFFEE QUALITY CLASSIFIER - SUITE DE PRUEBAS QA")
    print("=" * 70)
    
    # Configurar URL base
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🌐 URL de prueba: {base_url}")
    print()
    
    # Crear tester
    tester = CoffeeAPITester(base_url)
    
    # Ejecutar pruebas unitarias
    success = tester.run_all_tests()
    
    print("\n" + "=" * 70)
    
    # Ejecutar pruebas de integración
    integration_test()
    
    print("\n" + "=" * 70)
    print("✅ SUITE DE PRUEBAS COMPLETADA")
    
    if success:
        print("🎯 Resultado: TODAS LAS PRUEBAS PASARON")
        sys.exit(0)
    else:
        print("⚠️ Resultado: ALGUNAS PRUEBAS FALLARON")
        sys.exit(1)
