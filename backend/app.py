from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import numpy as np
import os
import glob
from scipy.optimize import differential_evolution
import traceback

app = Flask(__name__)

# -----------------------------
# PLANT MODEL MANAGER CLASS
# -----------------------------

class PlantModelManager:
    def __init__(self, dataset_dir="plant_datasets", model_dir="plant_models"):
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_info = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def get_available_plants(self):
        """Get list of available plant datasets"""
        csv_files = glob.glob(os.path.join(self.dataset_dir, "*_points.csv"))
        plant_names = []
        for file in csv_files:
            basename = os.path.basename(file)
            plant_name = basename.replace("_points.csv", "").replace("_", " ").title()
            plant_names.append(plant_name)
        return plant_names
    
    def get_dataset_path(self, plant_name):
        """Convert plant name to dataset file path"""
        safe_name = plant_name.lower().replace(" ", "_")
        return os.path.join(self.dataset_dir, f"{safe_name}_points.csv")
    
    def get_model_paths(self, plant_name):
        """Get model and scaler file paths for a plant"""
        safe_name = plant_name.lower().replace(" ", "_")
        model_path = os.path.join(self.model_dir, f"{safe_name}_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{safe_name}_scaler.pkl")
        return model_path, scaler_path
    
    def train_plant_model(self, plant_name, epochs=100, batch_size=16):
        """Train a neural network model for a specific plant"""
        dataset_path = self.get_dataset_path(plant_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found for {plant_name}: {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path, encoding='latin-1')
        df = df.drop(columns=["Species"])
        
        X = df.drop(columns=["Germination Rate (%)"])
        y = df["Germination Rate (%)"]
        
        # Store feature information
        self.feature_info[plant_name] = {
            'columns': X.columns.tolist(),
            'ranges': [(X[col].min(), X[col].max()) for col in X.columns]
        }
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build Neural Network
        model = models.Sequential([
            layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(16, activation="relu"),
            layers.Dense(1)  # regression output
        ])
        
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=0,
            validation_data=(X_test_scaled, y_test)
        )
        
        # Evaluate
        nn_preds = model.predict(X_test_scaled, verbose=0).flatten()
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, nn_preds)
        rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
        
        # Save model and scaler
        model_path, scaler_path = self.get_model_paths(plant_name)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        # Store in memory
        self.models[plant_name] = model
        self.scalers[plant_name] = scaler
        
        return {'r2': r2, 'rmse': rmse}
    
    def load_plant_model(self, plant_name):
        """Load a trained model for a specific plant"""
        model_path, scaler_path = self.get_model_paths(plant_name)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model files not found for {plant_name}")
        
        if plant_name not in self.models:
            self.models[plant_name] = tf.keras.models.load_model(model_path)
            self.scalers[plant_name] = joblib.load(scaler_path)
            
            # Load feature info from dataset
            dataset_path = self.get_dataset_path(plant_name)
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path, encoding='latin-1')
                df = df.drop(columns=["Species"])
                X = df.drop(columns=["Germination Rate (%)"])
                self.feature_info[plant_name] = {
                    'columns': X.columns.tolist(),
                    'ranges': [(X[col].min(), X[col].max()) for col in X.columns]
                }
    
    def predict_germination(self, plant_name, conditions):
        """Predict germination rate for given conditions"""
        if plant_name not in self.models:
            self.load_plant_model(plant_name)
        
        # Ensure conditions is a 2D array
        if len(np.array(conditions).shape) == 1:
            conditions = [conditions]
        
        # Convert to DataFrame with proper feature names
        feature_names = self.feature_info[plant_name]['columns']
        conditions_df = pd.DataFrame(conditions, columns=feature_names)
        conditions_scaled = self.scalers[plant_name].transform(conditions_df)
        
        predictions = self.models[plant_name].predict(conditions_scaled, verbose=0).flatten()
        return float(predictions[0]) if len(predictions) == 1 else [float(p) for p in predictions]
    
    def optimize_conditions(self, plant_name):
        """Find optimal conditions for maximum germination rate"""
        if plant_name not in self.models:
            self.load_plant_model(plant_name)
        
        feature_names = self.feature_info[plant_name]['columns']
        feature_ranges = self.feature_info[plant_name]['ranges']
        
        def objective_function(conditions):
            """Objective function to maximize germination rate"""
            return -self.predict_germination(plant_name, conditions)
        
        # Global optimization using differential evolution
        result = differential_evolution(
            objective_function,
            bounds=feature_ranges,
            seed=42,
            maxiter=1000,
            popsize=15,
            disp=False
        )
        
        optimal_conditions = result.x
        optimal_rate = -result.fun
        
        return {
            'plant_name': plant_name,
            'optimal_conditions': {name: float(value) for name, value in zip(feature_names, optimal_conditions)},
            'optimal_germination_rate': float(optimal_rate),
            'feature_names': feature_names,
            'feature_ranges': {name: {'min': float(r[0]), 'max': float(r[1])} for name, r in zip(feature_names, feature_ranges)}
        }
    
    def predict_all_plants(self, conditions):
        """Predict germination rates for all available plants"""
        available_plants = self.get_available_plants()
        predictions = {}
        
        for plant_name in available_plants:
            try:
                prediction = self.predict_germination(plant_name, conditions)
                predictions[plant_name] = prediction
            except Exception as e:
                predictions[plant_name] = f"Error: {str(e)}"
        
        return predictions

# Initialize the model manager
manager = PlantModelManager()

# -----------------------------
# API ROUTES
# -----------------------------

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        "message": "Plant Germination Prediction API",
        "endpoints": {
            "GET /": "API documentation",
            "GET /plants": "Get list of available plants",
            "POST /predict/<plant_name>": "Predict germination rate for specific plant",
            "POST /predict/all": "Predict germination rates for all plants",
            "GET /optimize/<plant_name>": "Get optimal conditions for specific plant",
            "GET /optimize/all": "Get optimal conditions for all plants",
            "POST /train/<plant_name>": "Train model for specific plant",
            "POST /train/all": "Train models for all plants"
        },
        "example_conditions": {
            "temperature": 25,
            "soil_moisture": 70,
            "altitude": 1000,
            "soil_ph": 6.5,
            "light_hours": 12
        }
    })

@app.route('/plants', methods=['GET'])
def get_plants():
    """Get list of available plants"""
    try:
        plants = manager.get_available_plants()
        return jsonify({
            "plants": plants,
            "count": len(plants)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<plant_name>', methods=['POST'])
def predict_single_plant(plant_name):
    """Predict germination rate for a specific plant"""
    try:
        data = request.get_json()
        
        # Expected format: {"conditions": [temp, moisture, altitude, pH, light]}
        if 'conditions' not in data:
            return jsonify({"error": "Missing 'conditions' in request body"}), 400
        
        conditions = data['conditions']
        
        # Validate conditions length (should be 5: temp, moisture, altitude, pH, light)
        if len(conditions) != 5:
            return jsonify({"error": "Conditions should have 5 values: [temperature, soil_moisture, altitude, soil_ph, light_hours]"}), 400
        
        prediction = manager.predict_germination(plant_name, conditions)
        
        return jsonify({
            "plant_name": plant_name,
            "conditions": {
                "temperature": conditions[0],
                "soil_moisture": conditions[1], 
                "altitude": conditions[2],
                "soil_ph": conditions[3],
                "light_hours": conditions[4]
            },
            "predicted_germination_rate": prediction
        })
        
    except FileNotFoundError as e:
        return jsonify({"error": f"Model not found for {plant_name}. Please train the model first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/all', methods=['POST'])
def predict_all_plants():
    """Predict germination rates for all plants"""
    try:
        data = request.get_json()
        
        if 'conditions' not in data:
            return jsonify({"error": "Missing 'conditions' in request body"}), 400
        
        conditions = data['conditions']
        
        if len(conditions) != 5:
            return jsonify({"error": "Conditions should have 5 values: [temperature, soil_moisture, altitude, soil_ph, light_hours]"}), 400
        
        predictions = manager.predict_all_plants(conditions)
        
        return jsonify({
            "conditions": {
                "temperature": conditions[0],
                "soil_moisture": conditions[1],
                "altitude": conditions[2], 
                "soil_ph": conditions[3],
                "light_hours": conditions[4]
            },
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize/<plant_name>', methods=['GET'])
def optimize_single_plant(plant_name):
    """Get optimal conditions for a specific plant"""
    try:
        result = manager.optimize_conditions(plant_name)
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({"error": f"Model not found for {plant_name}. Please train the model first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/optimize/all', methods=['GET'])
def optimize_all_plants():
    """Get optimal conditions for all plants"""
    try:
        plants = manager.get_available_plants()
        results = {}
        
        for plant_name in plants:
            try:
                result = manager.optimize_conditions(plant_name)
                results[plant_name] = result
            except Exception as e:
                results[plant_name] = {"error": str(e)}
        
        return jsonify({
            "optimization_results": results,
            "plant_count": len(plants)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train/<plant_name>', methods=['POST'])
def train_single_plant(plant_name):
    """Train model for a specific plant"""
    try:
        data = request.get_json() or {}
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 16)
        
        result = manager.train_plant_model(plant_name, epochs, batch_size)
        
        return jsonify({
            "message": f"Model trained successfully for {plant_name}",
            "plant_name": plant_name,
            "training_results": result
        })
        
    except FileNotFoundError as e:
        return jsonify({"error": f"Dataset not found for {plant_name}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train/all', methods=['POST'])
def train_all_plants():
    """Train models for all plants"""
    try:
        data = request.get_json() or {}
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 16)
        
        plants = manager.get_available_plants()
        results = {}
        
        for plant_name in plants:
            try:
                result = manager.train_plant_model(plant_name, epochs, batch_size)
                results[plant_name] = result
            except Exception as e:
                results[plant_name] = {"error": str(e)}
        
        return jsonify({
            "message": f"Training completed for {len(plants)} plants",
            "training_results": results,
            "plant_count": len(plants)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🌱 Plant Germination Prediction API Starting...")
    print("Available endpoints:")
    print("  GET  / - API documentation")
    print("  GET  /plants - List available plants")
    print("  POST /predict/<plant_name> - Predict for specific plant")
    print("  POST /predict/all - Predict for all plants")
    print("  GET  /optimize/<plant_name> - Optimize specific plant")
    print("  GET  /optimize/all - Optimize all plants")
    print("  POST /train/<plant_name> - Train specific plant")
    print("  POST /train/all - Train all plants")
    print("\n📍 API running on: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)