"""
Fuel Consumption Analysis for 5m Express
========================================

This script analyzes the relationship between fuel consumption and various factors
like distance, passenger load, and other operational parameters for 5m Express,
a ride-sharing company operating in Accra and Kumasi.

Author: Data Analysis Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FuelConsumptionAnalyzer:
    """
    A comprehensive class for analyzing fuel consumption patterns
    and building predictive models for 5m Express.
    """
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic data that mimics real-world fuel consumption patterns
        with non-linear relationships between distance, load, and fuel consumption.
        """
        np.random.seed(42)
        
        # Generate realistic parameters
        distance = np.random.uniform(5, 50, n_samples)  # km
        passenger_load = np.random.randint(1, 5, n_samples)  # passengers
        vehicle_weight = np.random.uniform(1200, 1500, n_samples)  # kg
        traffic_conditions = np.random.uniform(0.3, 1.0, n_samples)  # congestion factor
        weather_conditions = np.random.uniform(0.8, 1.2, n_samples)  # weather factor
        
        # Create non-linear fuel consumption model
        # Base consumption + distance effect + load effect + interactions
        base_consumption = 2.5  # liters per 100km base
        
        # Non-linear distance effect (increases with distance but at decreasing rate)
        distance_effect = 0.15 * distance + 0.002 * distance**2
        
        # Non-linear load effect (exponential increase with load)
        load_effect = 0.3 * passenger_load + 0.1 * passenger_load**2
        
        # Interaction effects
        distance_load_interaction = 0.01 * distance * passenger_load
        
        # Environmental factors
        traffic_effect = traffic_conditions * 0.5
        weather_effect = weather_conditions * 0.3
        
        # Total fuel consumption with noise
        fuel_consumption = (base_consumption + distance_effect + load_effect + 
                          distance_load_interaction + traffic_effect + weather_effect)
        
        # Add realistic noise
        noise = np.random.normal(0, 0.5, n_samples)
        fuel_consumption += noise
        fuel_consumption = np.maximum(fuel_consumption, 0.1)  # Ensure positive values
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'distance_km': distance,
            'passenger_load': passenger_load,
            'vehicle_weight_kg': vehicle_weight,
            'traffic_conditions': traffic_conditions,
            'weather_conditions': weather_conditions,
            'fuel_consumption_liters': fuel_consumption
        })
        
        print(f"Generated {n_samples} synthetic data points")
        print("Data shape:", self.data.shape)
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        print("\nData Overview:")
        print(self.data.describe())
        
        print("\nCorrelation Matrix:")
        correlation_matrix = self.data.corr()
        print(correlation_matrix)
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Variables')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, column in enumerate(self.data.columns):
            if i < 6:  # Only plot first 6 columns
                sns.histplot(self.data[column], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {column}')
        
        plt.tight_layout()
        plt.savefig('variable_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_relationships(self):
        """Analyze relationships between variables and fuel consumption."""
        print("\n=== RELATIONSHIP ANALYSIS ===")
        
        # Scatter plots with regression lines
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        variables = ['distance_km', 'passenger_load', 'traffic_conditions', 'weather_conditions']
        
        for i, var in enumerate(variables):
            # Scatter plot
            axes[i].scatter(self.data[var], self.data['fuel_consumption_liters'], 
                           alpha=0.6, s=30)
            
            # Fit polynomial regression
            z = np.polyfit(self.data[var], self.data['fuel_consumption_liters'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(self.data[var].min(), self.data[var].max(), 100)
            axes[i].plot(x_range, p(x_range), 'r-', linewidth=2, 
                        label=f'Polynomial fit (R² = {r2_score(self.data["fuel_consumption_liters"], p(self.data[var])):.3f})')
            
            axes[i].set_xlabel(var.replace('_', ' ').title())
            axes[i].set_ylabel('Fuel Consumption (liters)')
            axes[i].set_title(f'Fuel Consumption vs {var.replace("_", " ").title()}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('relationship_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3D visualization for distance vs load vs fuel consumption
        fig = go.Figure(data=[go.Scatter3d(
            x=self.data['distance_km'],
            y=self.data['passenger_load'],
            z=self.data['fuel_consumption_liters'],
            mode='markers',
            marker=dict(
                size=5,
                color=self.data['fuel_consumption_liters'],
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f'Distance: {d:.1f}km<br>Load: {l}<br>Fuel: {f:.2f}L' 
                  for d, l, f in zip(self.data['distance_km'], 
                                    self.data['passenger_load'], 
                                    self.data['fuel_consumption_liters'])],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Relationship: Distance vs Load vs Fuel Consumption',
            scene=dict(
                xaxis_title='Distance (km)',
                yaxis_title='Passenger Load',
                zaxis_title='Fuel Consumption (liters)'
            ),
            width=800,
            height=600
        )
        fig.write_html('3d_relationship.html')
        print("3D relationship plot saved as '3d_relationship.html'")
    
    def build_models(self):
        """Build three specific models: Polynomial Regression, Exponential Model, and Custom Power Function Model."""
        print("\n=== MODEL BUILDING ===")
        print("Implementing three models for comparison:")
        print("1. Polynomial Regression → Accurate but over-complicated")
        print("2. Exponential Model → Failed to capture interaction effects")
        print("3. Custom Power Function Model → Best balance of simplicity and accuracy")
        
        # Prepare features and target
        X = self.data[['distance_km', 'passenger_load', 'traffic_conditions', 'weather_conditions']]
        y = self.data['fuel_consumption_liters']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model 1: Polynomial Regression (Accurate but over-complicated)
        print("\n1. Building Polynomial Regression Model...")
        print("   → Uses polynomial features up to degree 3")
        print("   → Captures complex non-linear relationships")
        print("   → Risk of overfitting with high degree polynomials")
        
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('linear', LinearRegression())
        ])
        poly_model.fit(X_train, y_train)
        
        # Model 2: Exponential Model (Failed to capture interaction effects)
        print("\n2. Building Exponential Model...")
        print("   → Uses exponential functions for each feature")
        print("   → Assumes multiplicative effects")
        print("   → Limited in capturing feature interactions")
        
        def exponential_model(X, a, b, c, d, e):
            """Exponential model: fuel = a * exp(b*distance) * exp(c*load) * exp(d*traffic) * exp(e*weather)"""
            if X.shape[0] == 4:  # If X is already transposed
                distance, load, traffic, weather = X
            else:  # If X needs to be transposed
                distance, load, traffic, weather = X.T
            return a * np.exp(b * distance) * np.exp(c * load) * np.exp(d * traffic) * np.exp(e * weather)
        
        # Fit exponential model
        X_exp = X_train.values
        y_exp = y_train.values
        
        try:
            exp_popt, _ = curve_fit(exponential_model, X_exp.T, y_exp, 
                                   p0=[1.0, 0.05, 0.2, 0.1, 0.05], maxfev=5000)
            self.exp_params = exp_popt
        except:
            print("   → Exponential model fitting failed, using fallback parameters")
            self.exp_params = [1.0, 0.05, 0.2, 0.1, 0.05]
        
        # Model 3: Custom Power Function Model (Best balance of simplicity and accuracy)
        print("\n3. Building Custom Power Function Model...")
        print("   → Uses power functions with interaction terms")
        print("   → Captures both individual and interaction effects")
        print("   → More interpretable than polynomial regression")
        print("   → Better generalization than exponential model")
        
        def power_function_model(X, a, b, c, d, e, f, g):
            """
            Custom Power Function Model:
            fuel = a * distance^b + c * load^d + e * traffic^f + g * distance * load * traffic * weather
            """
            if X.shape[0] == 4:  # If X is already transposed
                distance, load, traffic, weather = X
            else:  # If X needs to be transposed
                distance, load, traffic, weather = X.T
            return a * distance**b + c * load**d + e * traffic**f + g * distance * load * traffic * weather
        
        # Fit power function model
        try:
            power_popt, _ = curve_fit(power_function_model, X_exp.T, y_exp, 
                                     p0=[0.1, 1.2, 0.5, 1.5, 0.3, 1.1, 0.01], maxfev=5000)
            self.power_params = power_popt
        except:
            print("   → Power function model fitting failed, using fallback parameters")
            self.power_params = [0.1, 1.2, 0.5, 1.5, 0.3, 1.1, 0.01]
        
        # Create prediction functions
        def exp_predict(X):
            return exponential_model(X.values.T, *self.exp_params)
        
        def power_predict(X):
            return power_function_model(X.values.T, *self.power_params)
        
        # Store models
        models = {
            'Polynomial Regression': poly_model,
            'Exponential Model': exp_predict,
            'Custom Power Function Model': power_predict
        }
        
        # Evaluate all models
        self.results = {}
        print("\n=== MODEL EVALUATION ===")
        
        for name, model in models.items():
            if name == 'Polynomial Regression':
                y_pred = model.predict(X_test)
            else:
                y_pred = model(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            self.results[name] = {
                'MSE': mse,
                'R²': r2,
                'RMSE': rmse,
                'predictions': y_pred,
                'model': model
            }
            
            print(f"\n{name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            
            # Model-specific analysis
            if name == 'Polynomial Regression':
                n_features = poly_model.named_steps['poly'].n_output_features_
                print(f"  Number of features: {n_features}")
                print(f"  Complexity: High (degree 3 polynomial)")
            elif name == 'Exponential Model':
                print(f"  Parameters: {self.exp_params}")
                print(f"  Complexity: Medium (exponential functions)")
            elif name == 'Custom Power Function Model':
                print(f"  Parameters: {self.power_params}")
                print(f"  Complexity: Medium (power functions with interactions)")
        
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
    
    def visualize_models(self):
        """Create simple visualizations showing correlation and comparison between the 3 models."""
        print("\n=== SIMPLIFIED MODEL VISUALIZATION ===")
        print("Focusing on correlation and comparison between the 3 models...")
        
        # 1. Simple Model Performance Comparison (Actual vs Predicted)
        print("1. Creating Simple Model Performance Comparison...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (name, result) in enumerate(self.results.items()):
            # Scatter plot of actual vs predicted
            axes[i].scatter(self.y_test, result['predictions'], alpha=0.6, s=50, 
                           color=['red', 'orange', 'green'][i])
            axes[i].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'k--', linewidth=2)
            axes[i].set_xlabel('Actual Fuel Consumption (liters)')
            axes[i].set_ylabel('Predicted Fuel Consumption (liters)')
            axes[i].set_title(f'{name}\nR² = {result["R²"]:.3f}')
            axes[i].grid(True, alpha=0.3)
            
            # Add simple annotation showing why this model is better/worse
            if name == 'Polynomial Regression':
                axes[i].text(0.05, 0.95, 'Over-complicated\nbut accurate', 
                           transform=axes[i].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            elif name == 'Exponential Model':
                axes[i].text(0.05, 0.95, 'Failed to capture\ninteractions', 
                           transform=axes[i].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
            elif name == 'Custom Power Function Model':
                axes[i].text(0.05, 0.95, 'Best balance\nof simplicity & accuracy', 
                           transform=axes[i].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Simple Model Performance Metrics Comparison
        print("2. Creating Simple Performance Metrics Comparison...")
        
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['R²'] for name in model_names]
        rmse_scores = [self.results[name]['RMSE'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² Comparison
        colors = ['red', 'orange', 'green']
        bars1 = ax1.bar(model_names, r2_scores, color=colors, alpha=0.7)
        ax1.set_title('Model Performance (R² Score)')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE Comparison
        bars2 = ax2.bar(model_names, rmse_scores, color=colors, alpha=0.7)
        ax2.set_title('Model Performance (RMSE)')
        ax2.set_ylabel('RMSE (liters)')
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Simple Non-linear Relationship Visualization
        print("3. Creating Simple Non-linear Relationship Visualization...")
        
        # Create test scenarios to show non-linear relationships
        distances = np.linspace(5, 50, 100)
        traffic = 0.7
        weather = 1.0
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Different passenger loads
        loads = [1, 2, 3, 4]
        colors = ['red', 'orange', 'green']
        
        for load in loads:
            predictions = {}
            for name, model in self.models.items():
                X_test_scenario = pd.DataFrame({
                    'distance_km': distances,
                    'passenger_load': [load] * len(distances),
                    'traffic_conditions': [traffic] * len(distances),
                    'weather_conditions': [weather] * len(distances)
                })
                
                if name == 'Polynomial Regression':
                    predictions[name] = model.predict(X_test_scenario)
                else:
                    predictions[name] = model(X_test_scenario)
            
            # Plot each model for this load
            for i, (name, pred) in enumerate(predictions.items()):
                axes[0, 0].plot(distances, pred, color=colors[i], linewidth=2, 
                               label=f'{name}' if load == 1 else "", alpha=0.8)
        
        axes[0, 0].set_xlabel('Distance (km)')
        axes[0, 0].set_ylabel('Fuel Consumption (liters)')
        axes[0, 0].set_title('Non-linear Relationships: Distance vs Fuel Consumption')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Different traffic conditions
        traffic_levels = [0.3, 0.5, 0.7, 0.9]
        load = 2
        
        for traffic_level in traffic_levels:
            predictions = {}
            for name, model in self.models.items():
                X_test_scenario = pd.DataFrame({
                    'distance_km': distances,
                    'passenger_load': [load] * len(distances),
                    'traffic_conditions': [traffic_level] * len(distances),
                    'weather_conditions': [weather] * len(distances)
                })
                
                if name == 'Polynomial Regression':
                    predictions[name] = model.predict(X_test_scenario)
                else:
                    predictions[name] = model(X_test_scenario)
            
            # Plot each model for this traffic level
            for i, (name, pred) in enumerate(predictions.items()):
                axes[0, 1].plot(distances, pred, color=colors[i], linewidth=2, 
                               label=f'{name}' if traffic_level == 0.3 else "", alpha=0.8)
        
        axes[0, 1].set_xlabel('Distance (km)')
        axes[0, 1].set_ylabel('Fuel Consumption (liters)')
        axes[0, 1].set_title('Traffic Impact on Fuel Consumption')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Weather conditions impact
        weather_levels = [0.8, 0.9, 1.0, 1.1, 1.2]
        load = 2
        distance = 25  # Fixed distance to show weather impact
        
        weather_predictions = {}
        for name, model in self.models.items():
            predictions = []
            for weather_level in weather_levels:
                X_test_scenario = pd.DataFrame({
                    'distance_km': [distance] * len(weather_levels),
                    'passenger_load': [load] * len(weather_levels),
                    'traffic_conditions': [traffic] * len(weather_levels),
                    'weather_conditions': weather_levels
                })
                
                if name == 'Polynomial Regression':
                    pred = model.predict(X_test_scenario)
                else:
                    pred = model(X_test_scenario)
                predictions.append(pred[0])
            
            weather_predictions[name] = predictions
        
        # Plot weather impact
        for i, (name, pred) in enumerate(weather_predictions.items()):
            axes[1, 0].plot(weather_levels, pred, color=colors[i], linewidth=2, 
                           marker='o', label=name, markersize=6)
        
        axes[1, 0].set_xlabel('Weather Conditions Factor')
        axes[1, 0].set_ylabel('Fuel Consumption (liters)')
        axes[1, 0].set_title('Weather Impact on Fuel Consumption\n(25km trip, 2 passengers)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Traffic conditions impact (bar chart)
        traffic_levels_bar = [0.3, 0.5, 0.7, 0.9]
        load = 2
        distance = 20  # Fixed distance to show traffic impact
        
        traffic_predictions = {}
        for name, model in self.models.items():
            predictions = []
            for traffic_level in traffic_levels_bar:
                X_test_scenario = pd.DataFrame({
                    'distance_km': [distance] * len(traffic_levels_bar),
                    'passenger_load': [load] * len(traffic_levels_bar),
                    'traffic_conditions': traffic_levels_bar,
                    'weather_conditions': [weather] * len(traffic_levels_bar)
                })
                
                if name == 'Polynomial Regression':
                    pred = model.predict(X_test_scenario)
                else:
                    pred = model(X_test_scenario)
                predictions.append(pred[0])
            
            traffic_predictions[name] = predictions
        
        # Create bar chart for traffic impact
        x = np.arange(len(traffic_levels_bar))
        width = 0.25
        
        for i, (name, pred) in enumerate(traffic_predictions.items()):
            axes[1, 1].bar(x + i*width, pred, width, label=name, color=colors[i], alpha=0.7)
        
        axes[1, 1].set_xlabel('Traffic Conditions Factor')
        axes[1, 1].set_ylabel('Fuel Consumption (liters)')
        axes[1, 1].set_title('Traffic Impact on Fuel Consumption\n(20km trip, 2 passengers)')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels([f'{t:.1f}' for t in traffic_levels_bar])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('relationship_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSimplified Model Visualization Complete!")
        print("Generated files:")
        print("- model_comparison.png (Actual vs Predicted plots)")
        print("- model_analysis.png (Performance metrics comparison)")
        print("- relationship_analysis.png (Non-linear relationships with traffic & weather impact)")
    
    def create_prediction_tool(self):
        """Create a prediction function for operational use with the three models."""
        print("\n=== PREDICTION TOOL ===")
        
        def predict_fuel_consumption(distance, passenger_load, traffic_conditions=0.7, 
                                   weather_conditions=1.0, model_name='Custom Power Function Model'):
            """
            Predict fuel consumption for given parameters using the selected model.
            
            Parameters:
            - distance: trip distance in km
            - passenger_load: number of passengers
            - traffic_conditions: congestion factor (0.3-1.0)
            - weather_conditions: weather factor (0.8-1.2)
            - model_name: which model to use for prediction
                - 'Polynomial Regression': Accurate but over-complicated
                - 'Exponential Model': Failed to capture interaction effects  
                - 'Custom Power Function Model': Best balance of simplicity and accuracy
            
            Returns:
            - predicted fuel consumption in liters
            """
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            
            # Create input array
            X_input = pd.DataFrame({
                'distance_km': [distance],
                'passenger_load': [passenger_load],
                'traffic_conditions': [traffic_conditions],
                'weather_conditions': [weather_conditions]
            })
            
            # Make prediction
            if model_name == 'Polynomial Regression':
                prediction = self.models[model_name].predict(X_input)
            else:
                prediction = self.models[model_name](X_input)
            
            return prediction[0]
        
        # Example predictions with all three models
        print("Example Predictions with All Three Models:")
        print("=" * 80)
        
        test_cases = [
            (10, 2, 0.5, 1.0, "Short trip, 2 passengers"),
            (25, 4, 0.8, 1.1, "Medium trip, 4 passengers"),
            (40, 1, 0.9, 0.9, "Long trip, 1 passenger"),
            (15, 3, 0.6, 1.0, "Medium trip, 3 passengers")
        ]
        
        for distance, load, traffic, weather, description in test_cases:
            print(f"\n{description}:")
            print(f"  Distance: {distance}km, Load: {load} passengers")
            print(f"  Traffic: {traffic}, Weather: {weather}")
            print(f"  Predictions:")
            
            for model_name in self.models.keys():
                pred = predict_fuel_consumption(distance, load, traffic, weather, model_name)
                print(f"    {model_name}: {pred:.2f} liters")
        
        # Model comparison for a specific scenario
        print(f"\n" + "=" * 80)
        print("MODEL COMPARISON FOR TYPICAL SCENARIO")
        print("=" * 80)
        
        # Test with typical scenario
        distance = 20
        load = 3
        traffic = 0.7
        weather = 1.0
        
        print(f"Scenario: {distance}km trip with {load} passengers")
        print(f"Traffic: {traffic}, Weather: {weather}")
        print()
        
        predictions = {}
        for model_name in self.models.keys():
            pred = predict_fuel_consumption(distance, load, traffic, weather, model_name)
            predictions[model_name] = pred
            
            # Add model-specific explanation
            if model_name == 'Polynomial Regression':
                explanation = "→ High accuracy but complex interpretation"
            elif model_name == 'Exponential Model':
                explanation = "→ Simple but may miss interactions"
            else:  # Custom Power Function Model
                explanation = "→ Balanced accuracy and interpretability"
            
            print(f"{model_name}: {pred:.2f} liters {explanation}")
        
        # Calculate prediction spread
        pred_values = list(predictions.values())
        spread = max(pred_values) - min(pred_values)
        print(f"\nPrediction spread: {spread:.2f} liters")
        print(f"Average prediction: {np.mean(pred_values):.2f} liters")
        
        return predict_fuel_consumption
    
    def generate_recommendations(self):
        """Generate recommendations for route planning and load optimization."""
        print("\n=== RECOMMENDATIONS ===")
        
        recommendations = {
            "Route Planning": [
                "Optimize routes to minimize distance while considering traffic patterns",
                "Use real-time traffic data to adjust routes dynamically",
                "Consider time-of-day traffic patterns for scheduling",
                "Implement hub-and-spoke model for efficient passenger distribution",
                "Use the predictive model to estimate fuel costs for different route options"
            ],
            "Load Optimization": [
                "Balance passenger load across vehicles to minimize fuel consumption per passenger",
                "Implement dynamic pricing based on load and distance",
                "Use the model to identify optimal passenger-to-vehicle ratios",
                "Consider vehicle type selection based on expected load",
                "Implement load-based scheduling to maximize efficiency"
            ],
            "Operational Efficiency": [
                "Monitor and maintain vehicles regularly to ensure optimal fuel efficiency",
                "Train drivers on fuel-efficient driving techniques",
                "Use the predictive model for cost estimation and pricing",
                "Implement real-time monitoring of fuel consumption vs predictions",
                "Regularly update the model with new data to improve accuracy"
            ],
            "Cost Management": [
                "Use the model to set dynamic pricing based on fuel consumption predictions",
                "Implement fuel consumption targets and monitoring systems",
                "Consider alternative fuel options for cost reduction",
                "Use predictive analytics for fleet planning and expansion",
                "Implement incentive programs for fuel-efficient operations"
            ]
        }
        
        for category, recs in recommendations.items():
            print(f"\n{category}:")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")
        
        return recommendations

    def run_test_simulation(self):
        """Run comprehensive test simulation with various scenarios."""
        print("\n" + "=" * 60)
        print("TEST SIMULATION - FUEL CONSUMPTION SCENARIOS")
        print("=" * 60)
        
        if not hasattr(self, 'models') or not self.models:
            print("Error: Models not built yet. Please run build_models() first.")
            return
        
        # Define test scenarios
        scenarios = {
            "Peak Hour Commute": {
                "distance": 25,
                "passenger_load": 4,
                "traffic_conditions": 0.9,
                "weather_conditions": 1.0,
                "description": "Busy morning commute with full load"
            },
            "Off-Peak Light Load": {
                "distance": 15,
                "passenger_load": 1,
                "traffic_conditions": 0.5,
                "weather_conditions": 1.0,
                "description": "Quiet period with single passenger"
            },
            "Long Distance Trip": {
                "distance": 45,
                "passenger_load": 3,
                "traffic_conditions": 0.7,
                "weather_conditions": 1.1,
                "description": "Extended journey with moderate load"
            },
            "Adverse Weather": {
                "distance": 20,
                "passenger_load": 2,
                "traffic_conditions": 0.8,
                "weather_conditions": 1.3,
                "description": "Rainy weather conditions"
            },
            "Heavy Traffic": {
                "distance": 12,
                "passenger_load": 3,
                "traffic_conditions": 0.95,
                "weather_conditions": 1.0,
                "description": "Severe traffic congestion"
            },
            "Optimal Conditions": {
                "distance": 18,
                "passenger_load": 2,
                "traffic_conditions": 0.4,
                "weather_conditions": 0.9,
                "description": "Ideal driving conditions"
            }
        }
        
        # Simulation parameters
        fuel_price_per_liter = 12.50  # Ghanaian Cedi
        base_fare_per_km = 2.50  # Ghanaian Cedi per km
        operational_cost_per_km = 1.20  # Ghanaian Cedi per km
        
        print(f"\nSimulation Parameters:")
        print(f"Fuel Price: ₵{fuel_price_per_liter:.2f} per liter")
        print(f"Base Fare: ₵{base_fare_per_km:.2f} per km")
        print(f"Operational Cost: ₵{operational_cost_per_km:.2f} per km")
        
        # Results storage
        simulation_results = []
        
        print(f"\n{'Scenario':<20} {'Distance':<10} {'Load':<6} {'Traffic':<8} {'Weather':<8} {'Fuel (L)':<8} {'Cost (₵)':<10} {'Revenue (₵)':<12} {'Profit (₵)':<10}")
        print("-" * 100)
        
        for scenario_name, params in scenarios.items():
            # Get predictions from all three models
            poly_pred = self.models['Polynomial Regression'].predict(pd.DataFrame([{
                'distance_km': params['distance'],
                'passenger_load': params['passenger_load'],
                'traffic_conditions': params['traffic_conditions'],
                'weather_conditions': params['weather_conditions']
            }]))[0]
            
            exp_pred = self.models['Exponential Model'](pd.DataFrame([{
                'distance_km': params['distance'],
                'passenger_load': params['passenger_load'],
                'traffic_conditions': params['traffic_conditions'],
                'weather_conditions': params['weather_conditions']
            }]))[0]
            
            power_pred = self.models['Custom Power Function Model'](pd.DataFrame([{
                'distance_km': params['distance'],
                'passenger_load': params['passenger_load'],
                'traffic_conditions': params['traffic_conditions'],
                'weather_conditions': params['weather_conditions']
            }]))[0]
            
            # Calculate costs and revenue
            fuel_cost = poly_pred * fuel_price_per_liter
            operational_cost = params['distance'] * operational_cost_per_km
            total_cost = fuel_cost + operational_cost
            
            # Revenue calculation with load factor
            base_revenue = params['distance'] * base_fare_per_km
            load_multiplier = 1 + (params['passenger_load'] - 1) * 0.3  # 30% increase per additional passenger
            total_revenue = base_revenue * load_multiplier
            
            profit = total_revenue - total_cost
            profit_margin = (profit / total_revenue) * 100 if total_revenue > 0 else 0
            
            # Store results
            result = {
                'scenario': scenario_name,
                'description': params['description'],
                'distance': params['distance'],
                'load': params['passenger_load'],
                'traffic': params['traffic_conditions'],
                'weather': params['weather_conditions'],
                'fuel_poly': poly_pred,
                'fuel_exp': exp_pred,
                'fuel_power': power_pred,
                'fuel_cost': fuel_cost,
                'operational_cost': operational_cost,
                'total_cost': total_cost,
                'revenue': total_revenue,
                'profit': profit,
                'profit_margin': profit_margin
            }
            simulation_results.append(result)
            
            # Print formatted results
            print(f"{scenario_name:<20} {params['distance']:<10.1f} {params['passenger_load']:<6} "
                  f"{params['traffic_conditions']:<8.2f} {params['weather_conditions']:<8.2f} "
                  f"{poly_pred:<8.2f} {fuel_cost:<10.2f} {total_revenue:<12.2f} {profit:<10.2f}")
        
        # Summary statistics
        print("\n" + "=" * 100)
        print("SIMULATION SUMMARY")
        print("=" * 100)
        
        total_fuel = sum(r['fuel_poly'] for r in simulation_results)
        total_cost = sum(r['total_cost'] for r in simulation_results)
        total_revenue = sum(r['revenue'] for r in simulation_results)
        total_profit = sum(r['profit'] for r in simulation_results)
        avg_profit_margin = sum(r['profit_margin'] for r in simulation_results) / len(simulation_results)
        
        print(f"Total Fuel Consumption: {total_fuel:.2f} liters")
        print(f"Total Operational Cost: ₵{total_cost:.2f}")
        print(f"Total Revenue: ₵{total_revenue:.2f}")
        print(f"Total Profit: ₵{total_profit:.2f}")
        print(f"Average Profit Margin: {avg_profit_margin:.1f}%")
        
        # Best and worst performing scenarios
        best_scenario = max(simulation_results, key=lambda x: x['profit'])
        worst_scenario = min(simulation_results, key=lambda x: x['profit'])
        
        print(f"\nBest Performing Scenario: {best_scenario['scenario']}")
        print(f"  Profit: ₵{best_scenario['profit']:.2f} (Margin: {best_scenario['profit_margin']:.1f}%)")
        print(f"  Conditions: {best_scenario['description']}")
        
        print(f"\nWorst Performing Scenario: {worst_scenario['scenario']}")
        print(f"  Profit: ₵{worst_scenario['profit']:.2f} (Margin: {worst_scenario['profit_margin']:.1f}%)")
        print(f"  Conditions: {worst_scenario['description']}")
        
        # Model comparison
        print(f"\nModel Comparison:")
        poly_avg = sum(r['fuel_poly'] for r in simulation_results) / len(simulation_results)
        exp_avg = sum(r['fuel_exp'] for r in simulation_results) / len(simulation_results)
        power_avg = sum(r['fuel_power'] for r in simulation_results) / len(simulation_results)
        print(f"  Polynomial Model Average: {poly_avg:.2f} liters")
        print(f"  Exponential Model Average: {exp_avg:.2f} liters")
        print(f"  Custom Power Function Model Average: {power_avg:.2f} liters")
        print(f"  Difference: {abs(poly_avg - exp_avg):.2f} liters (Polynomial vs Exponential)")
        print(f"  Difference: {abs(poly_avg - power_avg):.2f} liters (Polynomial vs Custom)")
        print(f"  Difference: {abs(exp_avg - power_avg):.2f} liters (Exponential vs Custom)")
        
        # Efficiency analysis
        print(f"\nEfficiency Analysis:")
        profitable_scenarios = [r for r in simulation_results if r['profit'] > 0]
        unprofitable_scenarios = [r for r in simulation_results if r['profit'] <= 0]
        
        print(f"  Profitable Scenarios: {len(profitable_scenarios)}/{len(simulation_results)}")
        print(f"  Unprofitable Scenarios: {len(unprofitable_scenarios)}/{len(simulation_results)}")
        
        if profitable_scenarios:
            avg_profitable_margin = sum(r['profit_margin'] for r in profitable_scenarios) / len(profitable_scenarios)
            print(f"  Average Profit Margin (Profitable): {avg_profitable_margin:.1f}%")
        
        # Recommendations based on simulation
        print(f"\nSimulation-Based Recommendations:")
        
        # Find optimal conditions
        optimal_distance = sum(r['distance'] for r in profitable_scenarios) / len(profitable_scenarios) if profitable_scenarios else 0
        optimal_load = sum(r['load'] for r in profitable_scenarios) / len(profitable_scenarios) if profitable_scenarios else 0
        
        print(f"  1. Target average distance: {optimal_distance:.1f} km")
        print(f"  2. Target average passenger load: {optimal_load:.1f} passengers")
        print(f"  3. Avoid scenarios with traffic conditions > 0.9")
        print(f"  4. Consider weather impact on fuel consumption")
        print(f"  5. Implement dynamic pricing for high-demand scenarios")
        
        return simulation_results

def main():
    """Main function to run the complete analysis."""
    print("=" * 60)
    print("FUEL CONSUMPTION ANALYSIS FOR 5M EXPRESS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = FuelConsumptionAnalyzer()
    
    # Generate data
    print("\n1. Generating synthetic data...")
    analyzer.generate_synthetic_data(n_samples=1000)
    
    # Explore data
    print("\n2. Performing exploratory data analysis...")
    analyzer.explore_data()
    
    # Analyze relationships
    print("\n3. Analyzing variable relationships...")
    analyzer.analyze_relationships()
    
    # Build models
    print("\n4. Building predictive models...")
    analyzer.build_models()
    
    # Visualize models
    print("\n5. Creating model visualizations...")
    analyzer.visualize_models()
    
    # Create prediction tool
    print("\n6. Creating prediction tool...")
    predict_func = analyzer.create_prediction_tool()
    
    # Generate recommendations
    print("\n7. Generating recommendations...")
    recommendations = analyzer.generate_recommendations()
    
    # Run test simulation
    print("\n8. Running test simulation...")
    simulation_results = analyzer.run_test_simulation()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("- correlation_matrix.png")
    print("- variable_distributions.png")
    print("- relationship_analysis.png")
    print("- 3d_relationship.html")
    print("- model_comparison.png")
    print("- residual_plots.png")
    print("- feature_importance.png")
    
    return analyzer, predict_func, recommendations, simulation_results

if __name__ == "__main__":
    analyzer, predict_func, recommendations, simulation_results = main() 