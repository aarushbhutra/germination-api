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
    
    def train_plant_model(self, plant_name, epochs=100, batch_size=16, verbose=0):
        """Train a neural network model for a specific plant"""
        dataset_path = self.get_dataset_path(plant_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found for {plant_name}: {dataset_path}")
        
        print(f"Training model for {plant_name}...")
        
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
            verbose=verbose,
            validation_data=(X_test_scaled, y_test),
            validation_split=0.1 if verbose == 0 else 0
        )
        
        # Evaluate
        nn_preds = model.predict(X_test_scaled, verbose=0).flatten()
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, nn_preds)
        rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
        
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # Save model and scaler
        model_path, scaler_path = self.get_model_paths(plant_name)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        # Store in memory
        self.models[plant_name] = model
        self.scalers[plant_name] = scaler
        
        print(f"  ✅ Model saved: {model_path}")
        print(f"  ✅ Scaler saved: {scaler_path}")
        
        return {'r2': r2, 'rmse': rmse, 'history': history}
    
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
        
        print(f"✅ Model loaded for {plant_name}")
    
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
        return predictions[0] if len(predictions) == 1 else predictions
    
    def optimize_conditions(self, plant_name, method='global', verbose=True):
        """Find optimal conditions for maximum germination rate"""
        if plant_name not in self.models:
            self.load_plant_model(plant_name)
        
        feature_names = self.feature_info[plant_name]['columns']
        feature_ranges = self.feature_info[plant_name]['ranges']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"OPTIMIZING CONDITIONS FOR {plant_name.upper()}")
            print(f"{'='*60}")
            print("Feature ranges:")
            for name, (min_val, max_val) in zip(feature_names, feature_ranges):
                print(f"  {name}: {min_val:.2f} - {max_val:.2f}")
        
        def objective_function(conditions):
            """Objective function to maximize germination rate"""
            return -self.predict_germination(plant_name, conditions)
        
        if method == 'global' or method == 'both':
            # Global optimization using differential evolution
            if verbose:
                print("\n🔍 Running global optimization...")
            
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
            
            if verbose:
                print(f"Maximum Germination Rate: {optimal_rate:.2f}%")
                print("Optimal Conditions:")
                for name, value in zip(feature_names, optimal_conditions):
                    print(f"  {name}: {value:.2f}")
            
            return {
                'method': 'global',
                'conditions': optimal_conditions,
                'germination_rate': optimal_rate,
                'feature_names': feature_names
            }
    
    def train_all_models(self, epochs=100, batch_size=16):
        """Train models for all available plants"""
        available_plants = self.get_available_plants()
        results = {}
        
        print(f"Training models for {len(available_plants)} plant species...")
        print(f"Available plants: {', '.join(available_plants)}")
        
        for plant_name in available_plants:
            try:
                result = self.train_plant_model(plant_name, epochs, batch_size, verbose=0)
                results[plant_name] = result
                print()
            except Exception as e:
                print(f"❌ Failed to train model for {plant_name}: {str(e)}")
                results[plant_name] = {'error': str(e)}
        
        return results
    
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

# -----------------------------
# MAIN EXECUTION FUNCTIONS
# -----------------------------

def main_single_plant(plant_name):
    """Run complete workflow for a single plant"""
    manager = PlantModelManager()
    
    # Train model
    manager.train_plant_model(plant_name)
    
    # Example prediction
    print(f"\n{'='*50}")
    print(f"EXAMPLE PREDICTION FOR {plant_name.upper()}")
    print(f"{'='*50}")
    
    # Use middle values as example
    feature_info = manager.feature_info[plant_name]
    example_conditions = [(r[0] + r[1]) / 2 for r in feature_info['ranges']]
    
    prediction = manager.predict_germination(plant_name, example_conditions)
    print(f"Example conditions: {example_conditions}")
    print(f"Predicted Germination Rate: {prediction:.2f}%")
    
    # Optimize conditions
    optimal_result = manager.optimize_conditions(plant_name)
    
    # Show improvement
    improvement = optimal_result['germination_rate'] - prediction
    print(f"\n🚀 Improvement: +{improvement:.2f} percentage points")
    print(f"Example: {prediction:.2f}% -> Optimal: {optimal_result['germination_rate']:.2f}%")

def main_all_plants():
    """Run complete workflow for all plants"""
    manager = PlantModelManager()
    
    # Train all models
    print("🌱 Training models for all plant species...")
    results = manager.train_all_models(epochs=100, batch_size=16)
    
    # Show training summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for plant_name, result in results.items():
        if 'error' in result:
            print(f"❌ {plant_name}: {result['error']}")
        else:
            print(f"✅ {plant_name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
    
    # Example: Compare predictions for same conditions across all plants
    print(f"\n{'='*60}")
    print("COMPARING ALL PLANTS WITH SAME CONDITIONS")
    print(f"{'='*60}")
    
    # Use moderate conditions
    example_conditions = [25, 70, 1000, 6.5, 12]  # temp, moisture, altitude, pH, light
    predictions = manager.predict_all_plants(example_conditions)
    
    print("Conditions: [Temp=25°C, Moisture=70%, Altitude=1000m, pH=6.5, Light=12h]")
    print("Predicted germination rates:")
    for plant_name, prediction in predictions.items():
        if isinstance(prediction, (int, float)):
            print(f"  {plant_name}: {prediction:.2f}%")
        else:
            print(f"  {plant_name}: {prediction}")

if __name__ == "__main__":
    # Choose what to run:
    
    # Option 1: Run for single plant
    # main_single_plant("Pinus Hartwegii")
    
    # Option 2: Run for all plants
    #main_all_plants()
    
    # Option 3: Interactive usage
    manager = PlantModelManager()
    available = manager.get_available_plants()
    print("Available plants:", available)
    # 
    # # Train specific plant
    # manager.train_plant_model("Brassica Juncea")
    # 
    # # Make prediction
    prediction = manager.predict_germination("Brassica Juncea", [20, 75, 500, 6.8, 12])
    print(f"Prediction: {prediction:.2f}%")
    # 
    # # Optimize
    optimal = manager.optimize_conditions("Brassica Juncea")