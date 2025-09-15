import csv
import numpy as np
import os

# -----------------------------
# PLANT CONFIGURATIONS
# -----------------------------

PLANT_CONFIGS = {
    "Pinus hartwegii": {
        "optimal": {
            "temp": 28,        # °C
            "moisture": 60,    # % field capacity
            "altitude": 1500,  # m
            "pH": 6.5,
            "light": 14        # hours/day
        },
        "ranges": {
            "temp": (10, 40),
            "moisture": (10, 100),
            "altitude": (300, 2500),
            "pH": (5.0, 8.5),
            "light": (4, 16)
        },
        "tolerances": {  # Standard deviations
            "temp": 3.5,
            "moisture": 12.5,
            "altitude": 900,
            "pH": 0.5,
            "light": 2
        }
    },
    "Gladiolus": {
        "optimal": {
            "temp": 22.5,      # °C
            "moisture": 70,    # % field capacity
            "altitude": 1900,  # m
            "pH": 6.0,
            "light": 10        # hours/day
        },
        "ranges": {
            "temp": (5, 45),
            "moisture": (20, 95),
            "altitude": (400, 3400),
            "pH": (4.5, 8.0),
            "light": (6, 16)
        },
        "tolerances": {
            "temp": 3.0,
            "moisture": 12.0,
            "altitude": 700,
            "pH": 0.5,
            "light": 2
        }
    },
    "Brassica juncea": {
        "optimal": {
            "temp": 25.5,
            "moisture": 75,
            "altitude": 1500,
            "pH": 6.5,
            "light": 14
        },
        "ranges": {
            "temp": (15, 35),
            "moisture": (40, 95),
            "altitude": (200, 3600),
            "pH": (5.5, 8.0),
            "light": (8, 16)
        },
        "tolerances": {
            "temp": 3.0,
            "moisture": 10.0,
            "altitude": 800,
            "pH": 0.5,
            "light": 2
        }
    },
    "Oryza sativa": {
        "optimal": {
            "temp": 27,
            "moisture": 90,
            "altitude": 500,
            "pH": 6.2,
            "light": 12
        },
        "ranges": {
            "temp": (20, 38),
            "moisture": (60, 100),
            "altitude": (0, 1500),
            "pH": (5.0, 7.5),
            "light": (10, 14)
        },
        "tolerances": {
            "temp": 4.0,
            "moisture": 8.0,
            "altitude": 400,
            "pH": 0.4,
            "light": 1
        }
    },
    "Rhododendron arboreum": {
        "optimal": {
            "temp": 15,
            "moisture": 65,
            "altitude": 2500,
            "pH": 5.8,
            "light": 8
        },
        "ranges": {
            "temp": (5, 25),
            "moisture": (40, 90),
            "altitude": (1500, 3700),
            "pH": (4.5, 6.5),
            "light": (4, 12)
        },
        "tolerances": {
            "temp": 2.5,
            "moisture": 10.0,
            "altitude": 600,
            "pH": 0.4,
            "light": 1.5
        }
    }
}

# -----------------------------
# FUNCTIONS
# -----------------------------

def generate_plant_dataset(species_name, n_points=17500, output_dir="plant_datasets"):
    """
    Generate a dataset for a specific plant species.
    
    Args:
        species_name (str): Name of the plant species (must be in PLANT_CONFIGS)
        n_points (int): Number of data points to generate
        output_dir (str): Directory to save the CSV file
    """
    
    if species_name not in PLANT_CONFIGS:
        raise ValueError(f"Plant '{species_name}' not found in configurations. Available: {list(PLANT_CONFIGS.keys())}")
    
    config = PLANT_CONFIGS[species_name]
    opt = config["optimal"]
    ranges = config["ranges"]
    tol = config["tolerances"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random conditions
    temperatures = np.random.uniform(*ranges["temp"], n_points)
    soil_moisture = np.random.uniform(*ranges["moisture"], n_points)
    altitudes = np.random.uniform(*ranges["altitude"], n_points)
    pH_levels = np.random.uniform(*ranges["pH"], n_points)
    light_hours = np.random.uniform(*ranges["light"], n_points)
    
    # Germination success modeled with multi-dimensional Gaussian
    germination_rates = np.exp(
        -((temperatures - opt["temp"]) ** 2) / (2 * tol["temp"] ** 2)
        -((soil_moisture - opt["moisture"]) ** 2) / (2 * tol["moisture"] ** 2)
        -((altitudes - opt["altitude"]) ** 2) / (2 * tol["altitude"] ** 2)
        -((pH_levels - opt["pH"]) ** 2) / (2 * tol["pH"] ** 2)
        -((light_hours - opt["light"]) ** 2) / (2 * tol["light"] ** 2)
    ) * 100
    
    # Clip between 0–100%
    germination_rates = np.clip(germination_rates, 0, 100)
    
    # Create filename
    safe_name = species_name.replace(" ", "_").lower()
    filename = os.path.join(output_dir, f"{safe_name}_points.csv")
    
    # Save to CSV
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Species",
            "Temperature (°C)",
            "Soil Moisture (%)",
            "Altitude (m)",
            "Soil pH",
            "Light Hours",
            "Germination Rate (%)"
        ])
        for t, m, a, pH, l, g in zip(temperatures, soil_moisture, altitudes, pH_levels, light_hours, germination_rates):
            writer.writerow([
                species_name,
                round(t, 2),
                round(m, 2),
                int(a),
                round(pH, 2),
                round(l, 2),
                round(g, 2)
            ])
    
    print(f"✅ File '{filename}' created with {n_points} data points for {species_name}.")
    return filename

def add_new_plant(species_name, optimal_conditions, condition_ranges, tolerances):
    """
    Add a new plant configuration to the system.
    
    Args:
        species_name (str): Name of the plant species
        optimal_conditions (dict): Optimal growing conditions
        condition_ranges (dict): Min/max ranges for each condition
        tolerances (dict): Standard deviations for tolerance
    """
    PLANT_CONFIGS[species_name] = {
        "optimal": optimal_conditions,
        "ranges": condition_ranges,
        "tolerances": tolerances
    }
    print(f"✅ Added configuration for {species_name}")

def list_available_plants():
    """List all available plant configurations."""
    print("Available plant species:")
    for plant in PLANT_CONFIGS.keys():
        print(f"  - {plant}")

def generate_all_datasets(n_points=17500):
    """Generate datasets for all configured plant species."""
    for species in PLANT_CONFIGS.keys():
        generate_plant_dataset(species, n_points)

# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":
    # Example usage:
    
    # List available plants
    list_available_plants()
    
    generate_all_datasets(n_points=20000)