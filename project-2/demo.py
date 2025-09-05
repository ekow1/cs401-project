#!/usr/bin/env python3
"""
Demo script for Lifecare Hospital ER Simulation
This script runs a quick test of the simulation with baseline parameters.
"""

from er_simulation import ERSimulation
import random

def run_demo():
    """Run a quick demo of the ER simulation"""
    print("LIFECARE HOSPITAL ER SIMULATION DEMO")
    print("="*50)
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Create and run simulation
    sim = ERSimulation()
    results = sim.run()
    
    # Print results
    sim.print_results()
    
    print("\nDemo completed successfully!")
    print("To run comprehensive analysis, execute: python scenario_analysis.py")

if __name__ == "__main__":
    run_demo() 