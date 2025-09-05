import random
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import statistics
import matplotlib.pyplot as plt
import numpy as np

class EventType(Enum):
    PATIENT_ARRIVAL = "patient_arrival"
    TRIAGE_COMPLETE = "triage_complete"
    CONSULTATION_COMPLETE = "consultation_complete"

@dataclass
class Event:
    time: float
    event_type: EventType
    patient_id: int
    priority: int = 0  # For priority queue
    
    def __lt__(self, other):
        return self.time < other.time

@dataclass
class Patient:
    id: int
    arrival_time: float
    triage_start_time: Optional[float] = None
    triage_end_time: Optional[float] = None
    consultation_start_time: Optional[float] = None
    consultation_end_time: Optional[float] = None
    
    @property
    def triage_wait_time(self) -> float:
        if self.triage_start_time is None:
            return 0
        return self.triage_start_time - self.arrival_time
    
    @property
    def triage_service_time(self) -> float:
        if self.triage_start_time is None or self.triage_end_time is None:
            return 0
        return self.triage_end_time - self.triage_start_time
    
    @property
    def consultation_wait_time(self) -> float:
        if self.consultation_start_time is None:
            return 0
        return self.consultation_start_time - self.triage_end_time
    
    @property
    def consultation_service_time(self) -> float:
        if self.consultation_start_time is None or self.consultation_end_time is None:
            return 0
        return self.consultation_end_time - self.consultation_start_time
    
    @property
    def total_wait_time(self) -> float:
        return self.triage_wait_time + self.consultation_wait_time
    
    @property
    def total_service_time(self) -> float:
        return self.triage_service_time + self.consultation_service_time
    
    @property
    def total_time_in_system(self) -> float:
        if self.consultation_end_time is None:
            return 0
        return self.consultation_end_time - self.arrival_time

class ERSimulation:
    def __init__(self, 
                 num_triage_nurses: int = 2,
                 num_doctors: int = 3,
                 shift_duration: float = 480.0,  # 8 hours in minutes
                 arrival_mean: float = 4.0,  # minutes
                 triage_min: float = 3.0,  # minutes
                 triage_max: float = 6.0,  # minutes
                 consultation_min: float = 5.0,  # minutes
                 consultation_max: float = 15.0):  # minutes
        
        # System parameters
        self.num_triage_nurses = num_triage_nurses
        self.num_doctors = num_doctors
        self.shift_duration = shift_duration
        self.arrival_mean = arrival_mean
        self.triage_min = triage_min
        self.triage_max = triage_max
        self.consultation_min = consultation_min
        self.consultation_max = consultation_max
        
        # Simulation state
        self.current_time = 0.0
        self.patient_id_counter = 0
        self.events = []  # Priority queue for events
        self.patients = {}  # Dictionary of all patients
        
        # Resource states
        self.available_triage_nurses = num_triage_nurses
        self.available_doctors = num_doctors
        
        # Queues
        self.triage_queue = []
        self.consultation_queue = []
        
        # Statistics tracking
        self.stats = {
            'total_patients': 0,
            'completed_patients': 0,
            'triage_wait_times': [],
            'consultation_wait_times': [],
            'total_wait_times': [],
            'triage_service_times': [],
            'consultation_service_times': [],
            'total_service_times': [],
            'triage_queue_lengths': [],
            'consultation_queue_lengths': [],
            'triage_utilization': [],
            'doctor_utilization': [],
            'time_points': []
        }
        
        # Initialize first patient arrival
        self._schedule_next_arrival()
    
    def _schedule_next_arrival(self):
        """Schedule the next patient arrival using exponential distribution"""
        if self.current_time < self.shift_duration:
            inter_arrival_time = random.expovariate(1.0 / self.arrival_mean)
            arrival_time = self.current_time + inter_arrival_time
            
            if arrival_time <= self.shift_duration:
                event = Event(arrival_time, EventType.PATIENT_ARRIVAL, self.patient_id_counter)
                heapq.heappush(self.events, event)
    
    def _generate_triage_time(self) -> float:
        """Generate triage service time using uniform distribution"""
        return random.uniform(self.triage_min, self.triage_max)
    
    def _generate_consultation_time(self) -> float:
        """Generate consultation service time using uniform distribution"""
        return random.uniform(self.consultation_min, self.consultation_max)
    
    def _update_statistics(self):
        """Update running statistics"""
        self.stats['time_points'].append(self.current_time)
        self.stats['triage_queue_lengths'].append(len(self.triage_queue))
        self.stats['consultation_queue_lengths'].append(len(self.consultation_queue))
        
        # Calculate utilization
        triage_util = (self.num_triage_nurses - self.available_triage_nurses) / self.num_triage_nurses
        doctor_util = (self.num_doctors - self.available_doctors) / self.num_doctors
        
        self.stats['triage_utilization'].append(triage_util)
        self.stats['doctor_utilization'].append(doctor_util)
    
    def _handle_patient_arrival(self, patient_id: int):
        """Handle a new patient arrival"""
        patient = Patient(id=patient_id, arrival_time=self.current_time)
        self.patients[patient_id] = patient
        self.stats['total_patients'] += 1
        self.patient_id_counter += 1
        
        # Try to start triage immediately
        if self.available_triage_nurses > 0:
            self._start_triage(patient)
        else:
            # Add to triage queue
            self.triage_queue.append(patient)
        
        # Schedule next arrival
        self._schedule_next_arrival()
    
    def _start_triage(self, patient: Patient):
        """Start triage for a patient"""
        self.available_triage_nurses -= 1
        patient.triage_start_time = self.current_time
        
        # Schedule triage completion
        triage_time = self._generate_triage_time()
        triage_end_time = self.current_time + triage_time
        
        event = Event(triage_end_time, EventType.TRIAGE_COMPLETE, patient.id)
        heapq.heappush(self.events, event)
    
    def _handle_triage_complete(self, patient_id: int):
        """Handle triage completion"""
        patient = self.patients[patient_id]
        patient.triage_end_time = self.current_time
        
        # Record triage statistics
        self.stats['triage_wait_times'].append(patient.triage_wait_time)
        self.stats['triage_service_times'].append(patient.triage_service_time)
        
        # Free up triage nurse
        self.available_triage_nurses += 1
        
        # Try to start consultation immediately
        if self.available_doctors > 0:
            self._start_consultation(patient)
        else:
            # Add to consultation queue
            self.consultation_queue.append(patient)
        
        # Check if there are patients waiting for triage
        if self.triage_queue and self.available_triage_nurses > 0:
            next_patient = self.triage_queue.pop(0)
            self._start_triage(next_patient)
    
    def _start_consultation(self, patient: Patient):
        """Start consultation for a patient"""
        self.available_doctors -= 1
        patient.consultation_start_time = self.current_time
        
        # Schedule consultation completion
        consultation_time = self._generate_consultation_time()
        consultation_end_time = self.current_time + consultation_time
        
        event = Event(consultation_end_time, EventType.CONSULTATION_COMPLETE, patient.id)
        heapq.heappush(self.events, event)
    
    def _handle_consultation_complete(self, patient_id: int):
        """Handle consultation completion"""
        patient = self.patients[patient_id]
        patient.consultation_end_time = self.current_time
        
        # Record consultation statistics
        self.stats['consultation_wait_times'].append(patient.consultation_wait_time)
        self.stats['consultation_service_times'].append(patient.consultation_service_time)
        self.stats['total_wait_times'].append(patient.total_wait_time)
        self.stats['total_service_times'].append(patient.total_service_time)
        
        self.stats['completed_patients'] += 1
        
        # Free up doctor
        self.available_doctors += 1
        
        # Check if there are patients waiting for consultation
        if self.consultation_queue and self.available_doctors > 0:
            next_patient = self.consultation_queue.pop(0)
            self._start_consultation(next_patient)
    
    def run(self):
        """Run the simulation"""
        print(f"Starting ER simulation...")
        print(f"Parameters: {self.num_triage_nurses} nurses, {self.num_doctors} doctors")
        print(f"Shift duration: {self.shift_duration} minutes")
        
        while self.events and self.current_time < self.shift_duration:
            # Get next event
            event = heapq.heappop(self.events)
            self.current_time = event.time
            
            # Update statistics
            self._update_statistics()
            
            # Handle event
            if event.event_type == EventType.PATIENT_ARRIVAL:
                self._handle_patient_arrival(event.patient_id)
            elif event.event_type == EventType.TRIAGE_COMPLETE:
                self._handle_triage_complete(event.patient_id)
            elif event.event_type == EventType.CONSULTATION_COMPLETE:
                self._handle_consultation_complete(event.patient_id)
        
        # Complete any remaining patients in the system
        self._complete_remaining_patients()
        
        print(f"Simulation completed at time {self.current_time:.2f} minutes")
        return self.get_results()
    
    def _complete_remaining_patients(self):
        """Complete all patients still in the system at shift end"""
        # Complete patients in triage queue (they will be counted as incomplete)
        for patient in self.triage_queue:
            if patient.triage_start_time is None:
                # Patient never started triage - count as incomplete
                pass
            else:
                # Patient was in triage but didn't complete - count as incomplete
                pass
        
        # Complete patients in consultation queue (they will be counted as incomplete)
        for patient in self.consultation_queue:
            if patient.consultation_start_time is None:
                # Patient never started consultation - count as incomplete
                pass
            else:
                # Patient was in consultation but didn't complete - count as incomplete
                pass
    
    def get_results(self) -> Dict:
        """Get simulation results and statistics"""
        results = {
            'total_patients': self.stats['total_patients'],
            'completed_patients': self.stats['completed_patients'],
            'avg_triage_wait_time': statistics.mean(self.stats['triage_wait_times']) if self.stats['triage_wait_times'] else 0,
            'avg_consultation_wait_time': statistics.mean(self.stats['consultation_wait_times']) if self.stats['consultation_wait_times'] else 0,
            'avg_total_wait_time': statistics.mean(self.stats['total_wait_times']) if self.stats['total_wait_times'] else 0,
            'avg_triage_service_time': statistics.mean(self.stats['triage_service_times']) if self.stats['triage_service_times'] else 0,
            'avg_consultation_service_time': statistics.mean(self.stats['consultation_service_times']) if self.stats['consultation_service_times'] else 0,
            'avg_total_service_time': statistics.mean(self.stats['total_service_times']) if self.stats['total_service_times'] else 0,
            'max_triage_queue_length': max(self.stats['triage_queue_lengths']) if self.stats['triage_queue_lengths'] else 0,
            'max_consultation_queue_length': max(self.stats['consultation_queue_lengths']) if self.stats['consultation_queue_lengths'] else 0,
            'avg_triage_utilization': statistics.mean(self.stats['triage_utilization']) if self.stats['triage_utilization'] else 0,
            'avg_doctor_utilization': statistics.mean(self.stats['doctor_utilization']) if self.stats['doctor_utilization'] else 0,
            'patients_per_hour': (self.stats['completed_patients'] / (self.shift_duration / 60)) if self.shift_duration > 0 else 0
        }
        
        return results
    
    def print_results(self):
        """Print formatted simulation results"""
        results = self.get_results()
        
        print("\n" + "="*60)
        print("ER SIMULATION RESULTS")
        print("="*60)
        print(f"Total Patients Arrived: {results['total_patients']}")
        print(f"Patients Completed: {results['completed_patients']}")
        print(f"Patients per Hour: {results['patients_per_hour']:.2f}")
        print()
        print("WAIT TIMES (minutes):")
        print(f"  Average Triage Wait: {results['avg_triage_wait_time']:.2f}")
        print(f"  Average Consultation Wait: {results['avg_consultation_wait_time']:.2f}")
        print(f"  Average Total Wait: {results['avg_total_wait_time']:.2f}")
        print()
        print("SERVICE TIMES (minutes):")
        print(f"  Average Triage Service: {results['avg_triage_service_time']:.2f}")
        print(f"  Average Consultation Service: {results['avg_consultation_service_time']:.2f}")
        print(f"  Average Total Service: {results['avg_total_service_time']:.2f}")
        print()
        print("QUEUE LENGTHS:")
        print(f"  Maximum Triage Queue: {results['max_triage_queue_length']}")
        print(f"  Maximum Consultation Queue: {results['max_consultation_queue_length']}")
        print()
        print("RESOURCE UTILIZATION:")
        print(f"  Average Triage Nurse Utilization: {results['avg_triage_utilization']:.2%}")
        print(f"  Average Doctor Utilization: {results['avg_doctor_utilization']:.2%}")
        print("="*60) 