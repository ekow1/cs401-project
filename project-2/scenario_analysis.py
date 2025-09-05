import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from er_simulation import ERSimulation
import random

class ScenarioAnalyzer:
    def __init__(self):
        self.scenarios = []
        self.results = []
    
    def run_baseline_scenario(self):
        """Run the baseline scenario with original parameters"""
        print("Running Baseline Scenario...")
        sim = ERSimulation()
        results = sim.run()
        sim.print_results()
        
        self.scenarios.append("Baseline (2 nurses, 3 doctors)")
        self.results.append(results)
        return results
    
    def run_staffing_scenarios(self):
        """Test different staffing configurations"""
        staffing_configs = [
            (1, 2, "1 nurse, 2 doctors"),
            (1, 3, "1 nurse, 3 doctors"),
            (2, 2, "2 nurses, 2 doctors"),
            (2, 3, "2 nurses, 3 doctors"),
            (2, 4, "2 nurses, 4 doctors"),
            (3, 3, "3 nurses, 3 doctors"),
            (3, 4, "3 nurses, 4 doctors"),
            (3, 5, "3 nurses, 5 doctors")
        ]
        
        print("\n" + "="*60)
        print("STAFFING SCENARIO ANALYSIS")
        print("="*60)
        
        for nurses, doctors, description in staffing_configs:
            print(f"\nTesting {description}...")
            sim = ERSimulation(num_triage_nurses=nurses, num_doctors=doctors)
            results = sim.run()
            
            self.scenarios.append(description)
            self.results.append(results)
    
    def run_shift_duration_scenarios(self):
        """Test different shift durations"""
        durations = [240, 360, 480, 600, 720]  # 4, 6, 8, 10, 12 hours
        
        print("\n" + "="*60)
        print("SHIFT DURATION SCENARIO ANALYSIS")
        print("="*60)
        
        for duration in durations:
            hours = duration / 60
            print(f"\nTesting {hours}-hour shift...")
            sim = ERSimulation(shift_duration=duration)
            results = sim.run()
            
            self.scenarios.append(f"{hours}-hour shift")
            self.results.append(results)
    
    def run_arrival_rate_scenarios(self):
        """Test different arrival rates"""
        arrival_means = [2, 3, 4, 5, 6]  # minutes between arrivals
        
        print("\n" + "="*60)
        print("ARRIVAL RATE SCENARIO ANALYSIS")
        print("="*60)
        
        for mean in arrival_means:
            print(f"\nTesting arrival rate: {mean} minutes between arrivals...")
            sim = ERSimulation(arrival_mean=mean)
            results = sim.run()
            
            self.scenarios.append(f"Arrival rate: {mean} min")
            self.results.append(results)
    
    def run_service_time_scenarios(self):
        """Test different service time distributions"""
        service_configs = [
            ((2, 4), (4, 12), "Faster service times"),
            ((3, 6), (5, 15), "Baseline service times"),
            ((4, 8), (6, 18), "Slower service times"),
            ((5, 10), (8, 20), "Much slower service times")
        ]
        
        print("\n" + "="*60)
        print("SERVICE TIME SCENARIO ANALYSIS")
        print("="*60)
        
        for (triage_min, triage_max), (consult_min, consult_max), description in service_configs:
            print(f"\nTesting {description}...")
            sim = ERSimulation(
                triage_min=triage_min,
                triage_max=triage_max,
                consultation_min=consult_min,
                consultation_max=consult_max
            )
            results = sim.run()
            
            self.scenarios.append(description)
            self.results.append(results)
    
    def create_comparison_table(self):
        """Create a comparison table of all scenarios"""
        if not self.results:
            print("No results to compare. Run scenarios first.")
            return
        
        # Create DataFrame
        data = []
        for i, (scenario, results) in enumerate(zip(self.scenarios, self.results)):
            data.append({
                'Scenario': scenario,
                'Total Patients': results['total_patients'],
                'Completed Patients': results['completed_patients'],
                'Patients/Hour': round(results['patients_per_hour'], 2),
                'Avg Total Wait (min)': round(results['avg_total_wait_time'], 2),
                'Avg Triage Wait (min)': round(results['avg_triage_wait_time'], 2),
                'Avg Consultation Wait (min)': round(results['avg_consultation_wait_time'], 2),
                'Max Triage Queue': results['max_triage_queue_length'],
                'Max Consultation Queue': results['max_consultation_queue_length'],
                'Triage Utilization': f"{results['avg_triage_utilization']:.1%}",
                'Doctor Utilization': f"{results['avg_doctor_utilization']:.1%}"
            })
        
        df = pd.DataFrame(data)
        return df
    
    def plot_wait_times_comparison(self):
        """Plot wait times comparison across scenarios"""
        if not self.results:
            print("No results to plot. Run scenarios first.")
            return
        
        scenarios = [s.replace(" (", "\n(") for s in self.scenarios]  # Break long names
        triage_waits = [r['avg_triage_wait_time'] for r in self.results]
        consultation_waits = [r['avg_consultation_wait_time'] for r in self.results]
        total_waits = [r['avg_total_wait_time'] for r in self.results]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width, triage_waits, width, label='Triage Wait', alpha=0.8)
        ax.bar(x, consultation_waits, width, label='Consultation Wait', alpha=0.8)
        ax.bar(x + width, total_waits, width, label='Total Wait', alpha=0.8)
        
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Wait Time (minutes)')
        ax.set_title('Average Wait Times by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wait_times_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_utilization_comparison(self):
        """Plot resource utilization comparison across scenarios"""
        if not self.results:
            print("No results to plot. Run scenarios first.")
            return
        
        scenarios = [s.replace(" (", "\n(") for s in self.scenarios]
        triage_util = [r['avg_triage_utilization'] * 100 for r in self.results]
        doctor_util = [r['avg_doctor_utilization'] * 100 for r in self.results]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width/2, triage_util, width, label='Triage Nurses', alpha=0.8)
        ax.bar(x + width/2, doctor_util, width, label='Doctors', alpha=0.8)
        
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Resource Utilization by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('utilization_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_queue_lengths(self):
        """Plot maximum queue lengths across scenarios"""
        if not self.results:
            print("No results to plot. Run scenarios first.")
            return
        
        scenarios = [s.replace(" (", "\n(") for s in self.scenarios]
        triage_queues = [r['max_triage_queue_length'] for r in self.results]
        consultation_queues = [r['max_consultation_queue_length'] for r in self.results]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width/2, triage_queues, width, label='Triage Queue', alpha=0.8)
        ax.bar(x + width/2, consultation_queues, width, label='Consultation Queue', alpha=0.8)
        
        ax.set_xlabel('Scenarios')
        ax.set_ylabel('Maximum Queue Length')
        ax.set_title('Maximum Queue Lengths by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('queue_lengths.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_bottleneck_analysis(self):
        """Visualize bottleneck in doctor consultations and effect of adding doctors"""
        if not self.results:
            print("No results to plot. Run scenarios first.")
            return
        staffing_indices = [i for i, s in enumerate(self.scenarios) if "nurse" in s.lower() and "doctor" in s.lower()]
        if not staffing_indices:
            print("No staffing scenario results found.")
            return

        # Collect data for both nurse and doctor bottleneck analysis
        nurse_counts = []
        doctor_counts = []
        avg_waits = []
        triage_queues = []
        consultation_queues = []
        for i in staffing_indices:
            desc = self.scenarios[i]
            import re
            match = re.search(r"(\d+) nurse.*?(\d+) doctor", desc)
            if match:
                nurse_counts.append(int(match.group(1)))
                doctor_counts.append(int(match.group(2)))
                avg_waits.append(self.results[i]['avg_total_wait_time'])
                triage_queues.append(self.results[i]['max_triage_queue_length'])
                consultation_queues.append(self.results[i]['max_consultation_queue_length'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Doctor bottleneck
        color = 'tab:blue'
        ax1.set_xlabel('Number of Doctors')
        ax1.set_ylabel('Avg Total Wait Time (min)', color=color)
        ax1.plot(doctor_counts, avg_waits, 'o-', color=color, label='Avg Wait Time')
        ax1.tick_params(axis='y', labelcolor=color)
        if 4 in doctor_counts:
            idx = doctor_counts.index(4)
            ax1.annotate(f"4 Doctors\nWait: {avg_waits[idx]:.2f} min",
                         (4, avg_waits[idx]),
                         textcoords="offset points", xytext=(0, -30), ha='center',
                         arrowprops=dict(arrowstyle="->", color='red'))
        ax1_2 = ax1.twinx()
        color2 = 'tab:red'
        ax1_2.set_ylabel('Max Consultation Queue', color=color2)
        ax1_2.plot(doctor_counts, consultation_queues, 's--', color=color2, label='Consultation Queue')
        ax1_2.tick_params(axis='y', labelcolor=color2)
        ax1.set_title('Doctor Consultation Bottleneck')

        # Nurse bottleneck
        color = 'tab:green'
        ax2.set_xlabel('Number of Triage Nurses')
        ax2.set_ylabel('Avg Total Wait Time (min)', color=color)
        ax2.plot(nurse_counts, avg_waits, 'o-', color=color, label='Avg Wait Time')
        ax2.tick_params(axis='y', labelcolor=color)
        if 2 in nurse_counts:
            idx = nurse_counts.index(2)
            ax2.annotate(f"2 Nurses\nWait: {avg_waits[idx]:.2f} min",
                         (2, avg_waits[idx]),
                         textcoords="offset points", xytext=(0, -30), ha='center',
                         arrowprops=dict(arrowstyle="->", color='orange'))
        ax2_2 = ax2.twinx()
        color2 = 'tab:purple'
        ax2_2.set_ylabel('Max Triage Queue', color=color2)
        ax2_2.plot(nurse_counts, triage_queues, 's--', color=color2, label='Triage Queue')
        ax2_2.tick_params(axis='y', labelcolor=color2)
        ax2.set_title('Triage Nurse Bottleneck')

        plt.suptitle('ER Bottleneck Analysis: Doctors vs Nurses')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('bottleneck_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Bottleneck analysis plot saved as 'bottleneck_analysis.png'.")
    
    def generate_recommendations(self):
        """Generate recommendations based on simulation results"""
        if not self.results:
            print("No results to analyze. Run scenarios first.")
            return
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS BASED ON SIMULATION RESULTS")
        print("="*60)
        
        # Find best performing scenarios
        best_wait_time = min(self.results, key=lambda x: x['avg_total_wait_time'])
        best_throughput = max(self.results, key=lambda x: x['patients_per_hour'])
        best_utilization = max(self.results, key=lambda x: x['avg_doctor_utilization'])
        
        print(f"\nBEST PERFORMING SCENARIOS:")
        print(f"Lowest Wait Time: {best_wait_time['avg_total_wait_time']:.2f} min")
        print(f"Highest Throughput: {best_throughput['patients_per_hour']:.2f} patients/hour")
        print(f"Highest Doctor Utilization: {best_utilization['avg_doctor_utilization']:.1%}")
        
        # Analyze bottlenecks
        baseline = self.results[0] if self.results else None
        if baseline:
            print(f"\nBOTTLENECK ANALYSIS (Baseline):")
            if baseline['avg_triage_wait_time'] > baseline['avg_consultation_wait_time']:
                print("Primary bottleneck: Triage nurses (longer wait times)")
                print("Recommendation: Add more triage nurses")
            else:
                print("Primary bottleneck: Doctors (longer wait times)")
                print("Recommendation: Add more doctors")
            
            print(f"\nQUEUE ANALYSIS:")
            print(f"Maximum triage queue: {baseline['max_triage_queue_length']} patients")
            print(f"Maximum consultation queue: {baseline['max_consultation_queue_length']} patients")
            
            if baseline['max_triage_queue_length'] > baseline['max_consultation_queue_length']:
                print("Triage queue is the limiting factor")
            else:
                print("Consultation queue is the limiting factor")
        
        # Staffing recommendations
        print(f"\nSTAFFING RECOMMENDATIONS:")
        
        # Find optimal staffing from scenarios
        staffing_results = [r for i, r in enumerate(self.results) 
                          if "nurse" in self.scenarios[i].lower() and "doctor" in self.scenarios[i].lower()]
        
        if staffing_results:
            best_staffing = min(staffing_results, key=lambda x: x['avg_total_wait_time'])
            print(f"Optimal staffing configuration: {best_staffing['avg_total_wait_time']:.2f} min average wait")
        
        print(f"\nGENERAL RECOMMENDATIONS:")
        print("1. Monitor queue lengths in real-time")
        print("2. Implement dynamic staffing based on demand")
        print("3. Consider extending shift hours during peak periods")
        print("4. Optimize triage process to reduce service time")
        print("5. Implement patient prioritization system")
        
        return {
            'best_wait_time': best_wait_time,
            'best_throughput': best_throughput,
            'best_utilization': best_utilization
        }

def main():
    """Run comprehensive scenario analysis"""
    analyzer = ScenarioAnalyzer()
    
    # Run all scenarios
    print("LIFECARE HOSPITAL ER SIMULATION ANALYSIS")
    print("="*60)
    
    # Baseline
    analyzer.run_baseline_scenario()
    
    # Staffing scenarios
    analyzer.run_staffing_scenarios()
    
    # Shift duration scenarios
    analyzer.run_shift_duration_scenarios()
    
    # Arrival rate scenarios
    analyzer.run_arrival_rate_scenarios()
    
    # Service time scenarios
    analyzer.run_service_time_scenarios()
    
    # Generate analysis
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Create comparison table
    df = analyzer.create_comparison_table()
    print("\nSCENARIO COMPARISON TABLE:")
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('scenario_comparison.csv', index=False)
    print("\nResults saved to 'scenario_comparison.csv'")
    
    # Generate plots
    print("\nGenerating plots...")
    analyzer.plot_wait_times_comparison()
    analyzer.plot_utilization_comparison()
    analyzer.plot_queue_lengths()
    
    # Generate recommendations
    analyzer.generate_recommendations()

    # Bottleneck visual analysis
    print("\nGenerating bottleneck visual analysis...")
    analyzer.plot_bottleneck_analysis()
    
    print("\nAnalysis complete! Check the generated files:")
    print("- scenario_comparison.csv")
    print("- wait_times_comparison.png")
    print("- utilization_comparison.png")
    print("- queue_lengths.png")

if __name__ == "__main__":
    main() 