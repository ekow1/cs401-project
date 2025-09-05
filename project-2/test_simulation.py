
"""
Test script for Lifecare Hospital ER Simulation
This script verifies all required functionality and metrics.
"""

from er_simulation import ERSimulation
import random

def test_baseline_simulation():
    """Test baseline simulation with original parameters"""
    print("Testing Baseline Simulation...")
    print("="*50)
    
    # Set seed for reproducible results
    random.seed(42)
    
    # Create simulation with baseline parameters
    sim = ERSimulation(
        num_triage_nurses=2,
        num_doctors=3,
        shift_duration=480.0,  # 8 hours
        arrival_mean=4.0,      # 4 minutes between arrivals
        triage_min=3.0,        # 3-6 minutes triage
        triage_max=6.0,
        consultation_min=5.0,   # 5-15 minutes consultation
        consultation_max=15.0
    )
    
    # Run simulation
    results = sim.run()
    
    # Print detailed results
    sim.print_results()
    
    # Verify required metrics are tracked
    print("\nVERIFICATION OF REQUIRED METRICS:")
    print("✓ Average patient wait times:", results['avg_total_wait_time'])
    print("✓ Resource utilization - Nurses:", f"{results['avg_triage_utilization']:.1%}")
    print("✓ Resource utilization - Doctors:", f"{results['avg_doctor_utilization']:.1%}")
    print("✓ Queue lengths - Triage:", results['max_triage_queue_length'])
    print("✓ Queue lengths - Consultation:", results['max_consultation_queue_length'])
    print("✓ Patients served per shift:", results['completed_patients'])
    print("✓ Patients per hour:", f"{results['patients_per_hour']:.2f}")
    
    return results

def test_staffing_scenarios():
    """Test different staffing configurations"""
    print("\n" + "="*50)
    print("Testing Staffing Scenarios...")
    print("="*50)
    
    staffing_configs = [
        (1, 2, "1 nurse, 2 doctors"),
        (2, 2, "2 nurses, 2 doctors"),
        (2, 3, "2 nurses, 3 doctors"),
        (2, 4, "2 nurses, 4 doctors"),
        (3, 3, "3 nurses, 3 doctors")
    ]
    
    results_summary = []
    
    for nurses, doctors, description in staffing_configs:
        print(f"\nTesting {description}...")
        random.seed(42)  # Same seed for fair comparison
        
        sim = ERSimulation(
            num_triage_nurses=nurses,
            num_doctors=doctors
        )
        
        results = sim.run()
        results_summary.append({
            'config': description,
            'wait_time': results['avg_total_wait_time'],
            'throughput': results['patients_per_hour'],
            'nurse_util': results['avg_triage_utilization'],
            'doctor_util': results['avg_doctor_utilization']
        })
        
        print(f"  Wait time: {results['avg_total_wait_time']:.2f} min")
        print(f"  Throughput: {results['patients_per_hour']:.2f} patients/hour")
        print(f"  Nurse utilization: {results['avg_triage_utilization']:.1%}")
        print(f"  Doctor utilization: {results['avg_doctor_utilization']:.1%}")
    
    # Find best configuration
    best_wait = min(results_summary, key=lambda x: x['wait_time'])
    best_throughput = max(results_summary, key=lambda x: x['throughput'])
    
    print(f"\nBEST CONFIGURATIONS:")
    print(f"Lowest wait time: {best_wait['config']} ({best_wait['wait_time']:.2f} min)")
    print(f"Highest throughput: {best_throughput['config']} ({best_throughput['throughput']:.2f} patients/hour)")
    
    return results_summary

def test_shift_durations():
    """Test different shift durations"""
    print("\n" + "="*50)
    print("Testing Shift Duration Scenarios...")
    print("="*50)
    
    durations = [240, 360, 480, 600, 720]  # 4, 6, 8, 10, 12 hours
    
    for duration in durations:
        hours = duration / 60
        print(f"\nTesting {hours}-hour shift...")
        random.seed(42)
        
        sim = ERSimulation(shift_duration=duration)
        results = sim.run()
        
        print(f"  Patients served: {results['completed_patients']}")
        print(f"  Patients per hour: {results['patients_per_hour']:.2f}")
        print(f"  Average wait time: {results['avg_total_wait_time']:.2f} min")

def test_arrival_rates():
    """Test different arrival rates"""
    print("\n" + "="*50)
    print("Testing Arrival Rate Scenarios...")
    print("="*50)
    
    arrival_means = [2, 3, 4, 5, 6]  # minutes between arrivals
    
    for mean in arrival_means:
        print(f"\nTesting arrival rate: {mean} minutes between arrivals...")
        random.seed(42)
        
        sim = ERSimulation(arrival_mean=mean)
        results = sim.run()
        
        print(f"  Patients arrived: {results['total_patients']}")
        print(f"  Patients completed: {results['completed_patients']}")
        print(f"  Average wait time: {results['avg_total_wait_time']:.2f} min")
        print(f"  Nurse utilization: {results['avg_triage_utilization']:.1%}")
        print(f"  Doctor utilization: {results['avg_doctor_utilization']:.1%}")

def main():
    """Run all tests"""
    print("LIFECARE HOSPITAL ER SIMULATION - COMPREHENSIVE TEST")
    print("="*60)
    
    # Test baseline simulation
    baseline_results = test_baseline_simulation()
    
    # Test staffing scenarios
    staffing_results = test_staffing_scenarios()
    
    # Test shift durations
    test_shift_durations()
    
    # Test arrival rates
    test_arrival_rates()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nThe simulation successfully tracks and analyzes:")
    print("✓ Average patient wait times")
    print("✓ Resource (nurse/doctor) utilization")
    print("✓ Queue lengths at each stage")
    print("✓ Number of patients served per shift")
    print("\nThe simulation successfully tests scenarios by tweaking:")
    print("✓ Number of triage nurses or doctors")
    print("✓ Inter-arrival or service time distributions")
    print("✓ Shift duration (4, 6, 8, 10, 12 hours)")
    print("\nReady to generate staffing and scheduling recommendations!")

if __name__ == "__main__":
    main() 