#!/usr/bin/env python3
"""
Power Function Model Analysis for Fuel Consumption
================================================
This script focuses exclusively on the Custom Power Function Model
for fuel consumption prediction with comprehensive visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class PowerFunctionAnalyzer:
    """Analyzer focused on Power Function model for fuel consumption prediction."""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.parameters = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.results = {}
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for fuel consumption analysis."""
        print("1. Generating synthetic data...")
        np.random.seed(42)
        
        # Generate features
        distance_km = np.random.uniform(5, 50, n_samples)
        passenger_load = np.random.randint(1, 5, n_samples)
        
        # Traffic conditions (light=1.0, moderate=1.2, heavy=1.5)
        traffic_categories = [1.0, 1.2, 1.5]
        traffic_weights = [0.4, 0.4, 0.2]
        traffic_factor = np.random.choice(traffic_categories, n_samples, p=traffic_weights)
        
        # Weather conditions (clear=1.0, rainy=1.3)
        weather_categories = [1.0, 1.3]
        weather_weights = [0.7, 0.3]
        weather_factor = np.random.choice(weather_categories, n_samples, p=weather_weights)
        
        # Calculate fuel consumption with realistic relationships
        base_consumption = 2.5
        distance_effect = 0.15 * distance_km + 0.002 * distance_km**2
        load_effect = 0.3 * passenger_load + 0.1 * passenger_load**2
        distance_load_interaction = 0.01 * distance_km * passenger_load
        traffic_effect = (traffic_factor - 1.0) * distance_km * 0.1
        weather_effect = (weather_factor - 1.0) * distance_km * 0.05
        
        fuel_liters = (base_consumption + distance_effect + load_effect + 
                      distance_load_interaction + traffic_effect + weather_effect)
        
        # Add realistic noise
        noise = np.random.normal(0, fuel_liters * 0.1, n_samples)
        fuel_liters += noise
        fuel_liters = np.maximum(fuel_liters, 0.1)
        
        self.data = pd.DataFrame({
            'distance_km': distance_km,
            'passenger_load': passenger_load,
            'traffic_factor': traffic_factor,
            'weather_factor': weather_factor,
            'fuel_liters': fuel_liters
        })
        
        print(f"Generated {n_samples} synthetic data points")
        print(f"Data shape: {self.data.shape}")
        
        # Show factor distributions
        print("\nTraffic Factor Distribution:")
        print(self.data['traffic_factor'].value_counts().sort_index())
        print("\nWeather Factor Distribution:")
        print(self.data['weather_factor'].value_counts().sort_index())
        
    def explore_data(self):
        """Perform basic data overview."""
        print("\n2. Performing data overview...")
        print("\n=== DATA OVERVIEW ===")
        
        # Data overview
        print("Data Overview:")
        print(self.data.describe())
        
        print("\nKey Parameters:")
        print("- Distance (km): Range from 5 to 50 km")
        print("- Passenger Load: 1 to 4 passengers")
        print("- Traffic Factor: Light (1.0), Moderate (1.2), Heavy (1.5)")
        print("- Weather Factor: Clear (1.0), Rainy (1.3)")
        print("- Target: Fuel Consumption (liters)")
        
    def power_function_model(self, X, a, b, c, d, e, f, g):
        """
        Custom Power Function Model:
        fuel = a * distance^b + c * load^d + e * traffic^f + g * distance * load * traffic * weather
        """
        distance_km = X['distance_km']
        passenger_load = X['passenger_load']
        traffic_factor = X['traffic_factor']
        weather_factor = X['weather_factor']
        
        return (a * distance_km**b + c * passenger_load**d + 
                e * traffic_factor**f + g * distance_km * passenger_load * 
                traffic_factor * weather_factor)
    
    def build_power_function_model(self):
        """Build and train the Power Function model."""
        print("\n3. Building Power Function Model...")
        print("=== POWER FUNCTION MODEL BUILDING ===")
        
        # Prepare data
        X = self.data[['distance_km', 'passenger_load', 'traffic_factor', 'weather_factor']]
        y = self.data['fuel_liters']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initial parameter guesses
        initial_guess = [0.1, 1.3, 0.3, 1.5, 2.5, 0.7, 0.01]
        
        try:
            # Fit the model
            self.parameters, _ = curve_fit(
                self.power_function_model, 
                self.X_train, 
                self.y_train,
                p0=initial_guess,
                maxfev=10000
            )
            
            # Create model function with fitted parameters
            self.model = lambda X: self.power_function_model(X, *self.parameters)
            
            # Make predictions
            self.predictions = self.model(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)
            rmse = np.sqrt(mse)
            
            self.results = {
                'MSE': mse,
                'R²': r2,
                'RMSE': rmse,
                'Parameters': self.parameters
            }
            
            print("Power Function Model Results:")
            print(f"  MSE: {mse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Parameters: {self.parameters}")
            print("  Model: fuel = a*distance^b + c*load^d + e*traffic^f + g*distance*load*traffic*weather")
            
        except Exception as e:
            print(f"Error fitting model: {e}")
            return None
    
    def create_comprehensive_visualizations(self):
        """Create individual visualizations: Actual vs Predicted, 3D Plot, and Error Distribution."""
        print("\n4. Creating individual model validation visualizations...")
        print("=== MODEL VALIDATION VISUALIZATIONS ===")
        
        # Set up the plotting style
        plt.style.use('default')
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. Actual vs Predicted Fuel Usage Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, self.predictions, alpha=0.6, color=colors[0], s=50)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Fuel Consumption (liters)', fontweight='bold', fontsize=12)
        plt.ylabel('Predicted Fuel Consumption (liters)', fontweight='bold', fontsize=12)
        plt.title('Actual vs Predicted Fuel Usage\nPower Function Model', 
                  fontweight='bold', fontsize=14, pad=15)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add R² text
        r2_text = f'R² = {self.results["R²"]:.4f}'
        plt.text(0.05, 0.95, r2_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('power_function_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 3D Plot: Distance, Load, Traffic vs Fuel Used (with Weather as Color)
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax3d = fig.add_subplot(111, projection='3d')
        
        # Sample data for 3D visualization (using test data)
        sample_size = min(200, len(self.X_test))  # Limit points for clarity
        indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        
        x = self.X_test.iloc[indices]['distance_km']
        y = self.X_test.iloc[indices]['passenger_load']
        z = self.X_test.iloc[indices]['traffic_factor']
        weather = self.X_test.iloc[indices]['weather_factor']
        fuel = self.y_test.iloc[indices]
        
        # Create scatter plot with weather as color
        scatter = ax3d.scatter(x, y, z, c=weather, cmap='RdYlBu', s=30, alpha=0.7)
        ax3d.set_xlabel('Distance (km)', fontweight='bold', fontsize=12)
        ax3d.set_ylabel('Passenger Load', fontweight='bold', fontsize=12)
        ax3d.set_zlabel('Traffic Factor', fontweight='bold', fontsize=12)
        ax3d.set_title('3D: Distance, Load, Traffic vs Fuel Consumption\n(Color = Weather Factor: Blue=Clear, Red=Rainy)', 
                       fontweight='bold', fontsize=14, pad=15)
        
        # Add colorbar for weather
        cbar = plt.colorbar(scatter, ax=ax3d, shrink=0.6, aspect=20)
        cbar.set_label('Weather Factor (1.0=Clear, 1.3=Rainy)', fontweight='bold', fontsize=12)
        
        # Add text annotation for weather interpretation
        ax3d.text2D(0.02, 0.98, 'Weather: Blue=Clear (1.0), Red=Rainy (1.3)', 
                    transform=ax3d.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('power_function_3d_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Error Distribution Graph (Model Accuracy)
        residuals = self.y_test - self.predictions
        
        plt.figure(figsize=(10, 8))
        plt.hist(residuals, bins=30, alpha=0.7, color=colors[2], edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Zero Error')
        plt.xlabel('Prediction Error (liters)', fontweight='bold', fontsize=12)
        plt.ylabel('Frequency', fontweight='bold', fontsize=12)
        plt.title('Error Distribution (Model Accuracy)\nPower Function Model', 
                  fontweight='bold', fontsize=14, pad=15)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add error statistics
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        error_text = f'Mean Error: {mean_error:.3f}\nStd Error: {std_error:.3f}'
        plt.text(0.02, 0.98, error_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                fontsize=10, fontweight='bold', verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('power_function_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Individual Model Validation Analysis Complete!")
        print("Generated files:")
        print("- power_function_actual_vs_predicted.png (Actual vs Predicted plot)")
        print("- power_function_3d_analysis.png (3D analysis plot)")
        print("- power_function_error_distribution.png (Error distribution plot)")
    
    def create_scenario_analysis(self):
        """Create clean scenario analysis with the Power Function model."""
        print("\n5. Creating scenario analysis...")
        print("=== SCENARIO ANALYSIS ===")
        
        # Define different scenarios
        scenarios = {
            'Short Urban Trip': {
                'distance': 10, 'load': 2, 'traffic': 1.2, 'weather': 1.0,
                'description': 'City center to suburbs, moderate traffic'
            },
            'Long Highway Trip': {
                'distance': 40, 'load': 3, 'traffic': 1.0, 'weather': 1.0,
                'description': 'Highway journey, light traffic, clear weather'
            },
            'Peak Hour Commute': {
                'distance': 25, 'load': 4, 'traffic': 1.5, 'weather': 1.0,
                'description': 'Rush hour travel, heavy traffic'
            },
            'Adverse Weather Trip': {
                'distance': 20, 'load': 2, 'traffic': 1.2, 'weather': 1.3,
                'description': 'Rainy weather conditions'
            },
            'Optimal Conditions': {
                'distance': 15, 'load': 1, 'traffic': 1.0, 'weather': 1.0,
                'description': 'Ideal driving conditions'
            }
        }
        
        # Calculate predictions for each scenario
        scenario_results = {}
        for name, params in scenarios.items():
            X_scenario = pd.DataFrame({
                'distance_km': [params['distance']],
                'passenger_load': [params['load']],
                'traffic_factor': [params['traffic']],
                'weather_factor': [params['weather']]
            })
            prediction = self.model(X_scenario)[0]
            scenario_results[name] = {
                'prediction': prediction,
                'params': params,
                'description': params['description']
            }
        
        # Create clean scenario comparison visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot scenario predictions
        scenario_names = list(scenario_results.keys())
        predictions = [scenario_results[name]['prediction'] for name in scenario_names]
        colors_scenario = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = ax.bar(scenario_names, predictions, color=colors_scenario, alpha=0.8)
        ax.set_ylabel('Predicted Fuel Consumption (liters)', fontweight='bold', fontsize=12)
        ax.set_title('Fuel Consumption by Scenario\nPower Function Model', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, predictions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}L', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('power_function_scenario_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print scenario results
        print("\nScenario Analysis Results:")
        print("=" * 60)
        for name, result in scenario_results.items():
            print(f"\n{name}:")
            print(f"  Description: {result['description']}")
            print(f"  Distance: {result['params']['distance']}km")
            print(f"  Load: {result['params']['load']} passengers")
            print(f"  Traffic: {result['params']['traffic']} (factor)")
            print(f"  Weather: {result['params']['weather']} (factor)")
            print(f"  Predicted Fuel: {result['prediction']:.2f} liters")
        
        print("\nScenario Analysis Complete!")
        print("Generated file: power_function_scenario_analysis.png")
    
    def run_analysis(self):
        """Run the complete Power Function analysis."""
        print("=" * 60)
        print("POWER FUNCTION MODEL ANALYSIS FOR FUEL CONSUMPTION")
        print("=" * 60)
        
        # Generate data
        self.generate_synthetic_data()
        
        # Explore data
        self.explore_data()
        
        # Build model
        self.build_power_function_model()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Create scenario analysis
        self.create_scenario_analysis()
        
        print("\n" + "=" * 60)
        print("POWER FUNCTION ANALYSIS COMPLETE")
        print("=" * 60)
        print("Generated files:")
        print("- power_function_actual_vs_predicted.png")
        print("- power_function_3d_analysis.png")
        print("- power_function_error_distribution.png")
        print("- power_function_scenario_analysis.png")
        
        return self.results

def main():
    """Main function to run the Power Function analysis."""
    analyzer = PowerFunctionAnalyzer()
    results = analyzer.run_analysis()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main() 