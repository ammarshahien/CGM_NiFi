import json
import logging
import random
import socket
import time
import threading
import argparse
import os
import signal
import sys
import math
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Enhanced Configuration with validation
@dataclass
class SensorConfig:
    device_id: str = 'CGM-SIM-001'
    patient_id: str = 'PATIENT-XYZ-789'
    update_interval_seconds: int = 1
    sensor_accuracy: float = 0.98  # Increased accuracy for more realistic readings
    calibration_drift: float = 0.05  # Reduced drift for stability
    patient_type: str = 'type1'  # type1, type2, prediabetes, gestational

@dataclass 
class CommunicationConfig:
    host: str = 'localhost'
    port: int = 9999
    timeout: int = 10
    max_retries: int = 5
    retry_delay: int = 5
    heartbeat_interval: int = 300

@dataclass
class PatientProfile:
    age: int = 35
    weight_kg: float = 70.0
    height_cm: float = 170.0
    diabetes_duration_years: int = 5
    hba1c: float = 7.2  # % (target <7% for most adults)
    insulin_sensitivity: float = 1.0  # Individual insulin sensitivity factor
    carb_ratio: float = 15.0  # grams of carbs per unit of insulin
    correction_factor: float = 50.0  # mg/dL drop per unit of insulin
    target_glucose_range: Tuple[float, float] = (70, 180)  # mg/dL
    # Added for realism
    dawn_phenomenon_strength: float = 1.0  # Individual variation in dawn phenomenon
    glucose_variability: float = 1.0  # Individual glucose variability factor

@dataclass
class GlucoseSimulationConfig:
    baseline_glucose: float = 120.0
    variability: float = 12.0  # Reduced for more stable readings
    meal_impact_duration: int = 180
    exercise_impact_duration: int = 60
    stress_impact_multiplier: float = 1.2
    dawn_phenomenon: bool = True
    dawn_effect_magnitude: float = 15.0  # Reduced dawn effect for realism
    
class GlucoseTrend(Enum):
    STEADY = "steady"
    RISING = "rising"
    RISING_RAPIDLY = "rising_rapidly"  
    FALLING = "falling"
    FALLING_RAPIDLY = "falling_rapidly"

class AlertLevel(Enum):
    NORMAL = "normal"
    LOW = "low"        # <70 mg/dL
    HIGH = "high"      # >180 mg/dL
    CRITICAL_LOW = "critical_low"   # <54 mg/dL
    CRITICAL_HIGH = "critical_high" # >250 mg/dL

class MealType(Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"

class ActivityType(Enum):
    REST = "rest"
    LIGHT_EXERCISE = "light_exercise"
    MODERATE_EXERCISE = "moderate_exercise"
    INTENSE_EXERCISE = "intense_exercise"
    SLEEP = "sleep"

@dataclass
class GlucoseReading:
    device_id: str
    patient_id: str
    timestamp: str
    glucose_level: float
    glucose_trend: str
    alert_level: str
    battery_level: int
    signal_strength: int
    calibration_status: str
    sensor_age_days: int
    temperature: float
    sequence_number: int
    
    # Enhanced dashboard features
    time_in_range_1h: float  # % in target range last hour
    time_in_range_24h: float  # % in target range last 24 hours
    estimated_hba1c: float  # Estimated A1C based on recent readings
    glucose_variability_cv: float  # Coefficient of variation (%)
    predicted_glucose_30min: float  # 30-minute prediction
    insulin_on_board: float  # Units of active insulin
    carbs_on_board: float  # Grams of active carbs
    current_activity: str  # Current activity state
    meal_status: str  # Recent meal information
    stress_level: float  # Current stress indicator (1.0-2.0)
    sleep_quality: float  # Sleep quality score (0-100)
    hydration_level: float  # Hydration estimate (0-100)

class MealEvent:
    def __init__(self, meal_type: MealType, carbs: float, timestamp: datetime):
        self.meal_type = meal_type
        self.carbs = carbs
        self.timestamp = timestamp
        self.duration_minutes = 240  # How long meal affects glucose
        # Individualize meal absorption based on meal type
        if meal_type == MealType.BREAKFAST:
            self.absorption_rate = random.uniform(0.8, 1.2)  # Breakfast often has faster carbs
        elif meal_type == MealType.LUNCH:
            self.absorption_rate = random.uniform(0.7, 1.1)
        elif meal_type == MealType.DINNER:
            self.absorption_rate = random.uniform(0.6, 1.0)  # Dinner often slower
        else:  # SNACK
            self.absorption_rate = random.uniform(0.9, 1.3)  # Snacks often fast

class InsulinDose:
    def __init__(self, units: float, timestamp: datetime, insulin_type: str = "rapid"):
        self.units = units
        self.timestamp = timestamp
        self.insulin_type = insulin_type
        self.duration_minutes = 300 if insulin_type == "rapid" else 1440  # 5h for rapid, 24h for long
        # Individual insulin sensitivity variation
        self.effectiveness = random.uniform(0.8, 1.2)

class EnhancedMockCGMSensor:
    def __init__(self, config_file: Optional[str] = None):
        self.load_config(config_file)
        self.running = False
        self.socket = None
        self.current_glucose = self.glucose_config.baseline_glucose
        self.sequence_number = 1
        self.sensor_start_time = datetime.now(timezone.utc)
        self.last_heartbeat = time.time()
        self.connection_attempts = 0
        
        # Enhanced tracking
        self.glucose_history = []  # Store last 24 hours of readings
        self.meal_events = []
        self.insulin_doses = []
        self.activity_state = ActivityType.REST
        self.current_stress = 1.0
        self.sleep_start = None
        self.hydration = 80.0  # Start at 80%
        
        # Physiological state - enhanced for realism
        self.last_meal_time = None
        self.last_exercise_time = None
        self.sleep_cycle_offset = random.uniform(0, 24)
        self.individual_variability = random.uniform(0.8, 1.2)  # Individual metabolic rate
        
        # Sensor characteristics
        self.calibration_drift = random.uniform(-0.1, 0.1)  # Small initial drift
        self.battery_drain_rate = random.uniform(0.08, 0.15)  # More realistic battery drain
        self.initial_battery = random.randint(95, 100)  # Start with nearly full battery
        
        # Dawn phenomenon timing - individual variation
        self.dawn_start_hour = random.uniform(3, 5)
        self.dawn_end_hour = random.uniform(7, 9)
        self.dawn_strength = self.patient_profile.dawn_phenomenon_strength
        
        # Glucose dynamics
        self.glucose_momentum = 0.85  # Higher momentum for more realistic gradual changes
        self.last_glucose_change = 0
        
        self.setup_logging()
        self.setup_signal_handlers()

    def load_config(self, config_file: Optional[str]):
        """Load configuration from file or use defaults"""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                self.sensor_config = SensorConfig(**config_data.get('sensor', {}))
                self.comm_config = CommunicationConfig(**config_data.get('communication', {}))
                self.glucose_config = GlucoseSimulationConfig(**config_data.get('glucose_simulation', {}))
                self.patient_profile = PatientProfile(**config_data.get('patient_profile', {}))
            except Exception as e:
                logging.warning(f"Failed to load config file: {e}. Using defaults.")
                self._use_default_config()
        else:
            self._use_default_config()

    def _use_default_config(self):
        self.sensor_config = SensorConfig()
        self.comm_config = CommunicationConfig()
        self.glucose_config = GlucoseSimulationConfig()
        self.patient_profile = PatientProfile()

    def setup_logging(self):
        """Enhanced logging with file output"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(
            log_dir / f"cgm_sensor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Create logger and add handlers
        self.logger = logging.getLogger('EnhancedMockCGMSensor')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False

    def setup_signal_handlers(self):
        """Setup graceful shutdown on SIGINT/SIGTERM"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}. Initiating shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def simulate_daily_schedule(self):
        """Simulate realistic daily activities and meals with more natural timing"""
        current_time = datetime.now()
        hour = current_time.hour
        minute = current_time.minute
        time_of_day = hour + minute/60
        
        # Breakfast time (6:30-9:00 AM) - more likely around 7-8 AM
        if 6.5 <= time_of_day <= 9.0:
            breakfast_prob = 0.005 * math.exp(-((time_of_day - 7.5) ** 2) / 0.5)  # Gaussian distribution
            if random.random() < breakfast_prob:
                self.add_meal_event(MealType.BREAKFAST, random.uniform(30, 60))
            
        # Lunch time (11:30 AM-2:00 PM) - peak around 12-1 PM
        elif 11.5 <= time_of_day <= 14.0:
            lunch_prob = 0.005 * math.exp(-((time_of_day - 12.5) ** 2) / 0.5)
            if random.random() < lunch_prob:
                self.add_meal_event(MealType.LUNCH, random.uniform(40, 80))
            
        # Dinner time (5:00-8:00 PM) - peak around 6-7 PM
        elif 17.0 <= time_of_day <= 20.0:
            dinner_prob = 0.005 * math.exp(-((time_of_day - 18.5) ** 2) / 0.5)
            if random.random() < dinner_prob:
                self.add_meal_event(MealType.DINNER, random.uniform(50, 90))
            
        # Snacks (random throughout day) - more likely mid-morning and mid-afternoon
        elif (10.0 <= time_of_day <= 11.0) or (15.0 <= time_of_day <= 16.0):
            if random.random() < 0.003:
                self.add_meal_event(MealType.SNACK, random.uniform(10, 30))
        
        # Exercise patterns - more likely in morning or evening
        if (6.0 <= hour <= 8.0) or (17.0 <= hour <= 20.0):
            if random.random() < 0.002:
                self.start_exercise()
        
        # Sleep detection (10 PM - 6 AM) with transition periods
        if 22 <= hour or hour <= 6:
            if self.activity_state != ActivityType.SLEEP and random.random() < 0.1:
                self.activity_state = ActivityType.SLEEP
                self.sleep_start = current_time
        else:
            if self.activity_state == ActivityType.SLEEP and random.random() < 0.3:
                self.activity_state = ActivityType.REST

    def add_meal_event(self, meal_type: MealType, carbs: float):
        """Add a meal event and simulate insulin dosing with more realistic timing"""
        current_time = datetime.now(timezone.utc)
        meal = MealEvent(meal_type, carbs, current_time)
        self.meal_events.append(meal)
        
        # Simulate insulin dosing based on carb ratio with more realistic timing
        if self.sensor_config.patient_type in ['type1', 'gestational']:
            insulin_units = carbs / self.patient_profile.carb_ratio
            # Add realistic variation based on meal type and individual factors
            insulin_units *= random.uniform(0.85, 1.15) * self.individual_variability
            
            # Add some timing variation (bolus might be before, during, or after meal)
            dose_time = current_time + timedelta(minutes=random.randint(-15, 30))
            self.insulin_doses.append(InsulinDose(insulin_units, dose_time))
            
        self.last_meal_time = current_time
        self.logger.info(f"Meal event: {meal_type.value} - {carbs:.1f}g carbs")

    def start_exercise(self):
        """Start exercise session with more realistic intensity distribution"""
        self.last_exercise_time = datetime.now(timezone.utc)
        
        # Most exercise is light or moderate, intense is rare
        exercise_weights = [0.4, 0.4, 0.15, 0.05]  # light, moderate, intense, rest (shouldn't happen)
        exercise_types = [ActivityType.LIGHT_EXERCISE, ActivityType.MODERATE_EXERCISE, 
                         ActivityType.INTENSE_EXERCISE, ActivityType.REST]
        
        self.activity_state = random.choices(exercise_types, weights=exercise_weights, k=1)[0]
        self.logger.info(f"Exercise started: {self.activity_state.value}")

    def calculate_realistic_glucose_effects(self):
        """Calculate realistic glucose effects based on physiology with improved modeling"""
        current_time = datetime.now(timezone.utc)
        total_effect = 0
        active_insulin = 0
        
        # Meal effects with more realistic absorption curves
        for meal in self.meal_events[:]:
            time_since_meal = (current_time - meal.timestamp).total_seconds() / 60
            if time_since_meal > meal.duration_minutes:
                self.meal_events.remove(meal)
                continue
                
            # More realistic carb absorption curve with individual variation
            if time_since_meal < 30:
                # Initial rise (0-30 min)
                peak_effect = meal.carbs * 2.5 * meal.absorption_rate
                effect = peak_effect * (time_since_meal / 30) * 0.4
            elif time_since_meal < 90:
                # Peak effect (30-90 min)
                peak_effect = meal.carbs * 2.5 * meal.absorption_rate
                effect = peak_effect * 0.4 + peak_effect * 0.6 * math.sin((time_since_meal - 30) * math.pi / 120)
            else:
                # Declining effect (90+ min)
                remaining_time = meal.duration_minutes - time_since_meal
                effect = (meal.carbs * 1.0 * meal.absorption_rate) * (remaining_time / (meal.duration_minutes - 90))
            
            total_effect += effect * self.individual_variability

        # Insulin effects with more realistic pharmacokinetics
        for dose in self.insulin_doses[:]:
            time_since_dose = (current_time - dose.timestamp).total_seconds() / 60
            if time_since_dose > dose.duration_minutes:
                self.insulin_doses.remove(dose)
                continue
                
            # More realistic insulin activity curve
            if dose.insulin_type == "rapid":
                # Four-phase model: delay, rise, peak, decline
                if time_since_dose < 15:
                    activity = 0.05  # Initial delay
                elif time_since_dose < 75:
                    # Rising phase (15-75 min)
                    activity = 0.05 + 0.85 * ((time_since_dose - 15) / 60)
                elif time_since_dose < 180:
                    # Peak phase (75-180 min)
                    activity = 0.9 - 0.2 * ((time_since_dose - 75) / 105)
                else:
                    # Declining phase (180+ min)
                    activity = 0.7 * (dose.duration_minutes - time_since_dose) / (dose.duration_minutes - 180)
                
                glucose_drop = dose.units * self.patient_profile.correction_factor * activity * dose.effectiveness / 100
                total_effect -= glucose_drop
                active_insulin += dose.units * activity * dose.effectiveness
        
        # Exercise effects with more gradual onset and recovery
        if self.last_exercise_time:
            time_since_exercise = (current_time - self.last_exercise_time).total_seconds() / 60
            if time_since_exercise <= 180:  # 3 hour effect with gradual decline
                intensity_multiplier = {
                    ActivityType.LIGHT_EXERCISE: 0.5,
                    ActivityType.MODERATE_EXERCISE: 1.0,
                    ActivityType.INTENSE_EXERCISE: 1.8
                }.get(self.activity_state, 0)
                
                # Exercise effect follows a curve: gradual increase, then gradual decrease
                if time_since_exercise < 30:
                    # Building effect
                    exercise_effect = -15 * intensity_multiplier * (time_since_exercise / 30)
                else:
                    # Declining effect
                    exercise_effect = -15 * intensity_multiplier * (1 - (time_since_exercise - 30) / 150)
                
                total_effect += exercise_effect

        return total_effect, active_insulin

    def calculate_circadian_effects(self):
        """Enhanced circadian rhythm simulation with more realistic patterns"""
        current_time = datetime.now()
        current_hour = current_time.hour + current_time.minute / 60
        day_of_week = current_time.weekday()
        
        # Dawn phenomenon (individual variation)
        dawn_effect = 0
        if self.glucose_config.dawn_phenomenon:
            if self.dawn_start_hour <= current_hour <= self.dawn_end_hour:
                # Gaussian curve for dawn effect with individual variation
                dawn_center = (self.dawn_start_hour + self.dawn_end_hour) / 2
                dawn_width = (self.dawn_end_hour - self.dawn_start_hour) / 3
                dawn_effect = (self.glucose_config.dawn_effect_magnitude * self.dawn_strength * 
                             math.exp(-((current_hour - dawn_center) ** 2) / (2 * dawn_width ** 2)))
        
        # Cortisol rhythm effects - higher on weekdays
        weekday_factor = 1.2 if day_of_week < 5 else 0.8  # Higher on weekdays
        cortisol_effect = 5 * math.sin((current_hour - 8) * math.pi / 12) * weekday_factor
        
        # Sleep effect - more pronounced during deep sleep (typically 1-3 AM)
        if self.activity_state == ActivityType.SLEEP:
            # Maximum effect around 2 AM
            sleep_effect = -8 * math.exp(-((current_hour - 2) ** 2) / 2)
        else:
            sleep_effect = 0
            
        return dawn_effect + cortisol_effect + sleep_effect

    def calculate_time_in_range(self, hours: int = 24) -> float:
        """Calculate time in range for dashboard"""
        if len(self.glucose_history) < 2:
            return 100.0  # Default if insufficient data
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_readings = [
            reading for reading in self.glucose_history 
            if datetime.fromisoformat(reading['timestamp'].replace('Z', '+00:00')) >= cutoff_time
        ]
        
        if not recent_readings:
            return 100.0
            
        in_range_count = sum(
            1 for reading in recent_readings
            if self.patient_profile.target_glucose_range[0] <= reading['glucose_level'] <= self.patient_profile.target_glucose_range[1]
        )
        
        return (in_range_count / len(recent_readings)) * 100

    def calculate_glucose_variability(self) -> float:
        """Calculate coefficient of variation for glucose variability"""
        if len(self.glucose_history) < 10:
            return 20.0  # Default CV
            
        recent_glucose = [reading['glucose_level'] for reading in self.glucose_history[-288:]]  # Last 24 hours
        if len(recent_glucose) < 2:
            return 20.0
            
        mean_glucose = sum(recent_glucose) / len(recent_glucose)
        variance = sum((x - mean_glucose) ** 2 for x in recent_glucose) / len(recent_glucose)
        std_dev = math.sqrt(variance)
        
        cv = (std_dev / mean_glucose) * 100 if mean_glucose > 0 else 20.0
        return min(cv, 100.0)  # Cap at 100%

    def estimate_hba1c(self) -> float:
        """Estimate HbA1c from average glucose (ADAG formula)"""
        if len(self.glucose_history) < 144:  # Need at least 12 hours of data
            return self.patient_profile.hba1c
            
        # Use last 2 weeks of data if available
        recent_readings = self.glucose_history[-2016:] if len(self.glucose_history) >= 2016 else self.glucose_history
        avg_glucose = sum(reading['glucose_level'] for reading in recent_readings) / len(recent_readings)
        
        # ADAG formula: HbA1c (%) = (avg_glucose + 46.7) / 28.7
        estimated_a1c = (avg_glucose + 46.7) / 28.7
        return round(estimated_a1c, 1)

    def predict_glucose_30min(self) -> float:
        """Improved 30-minute glucose prediction based on trend and momentum"""
        if len(self.glucose_history) < 6:  # Need at least 30 minutes of data
            return self.current_glucose
            
        # Calculate trend from last 6 readings (30 minutes)
        recent_readings = self.glucose_history[-6:]
        if len(recent_readings) < 2:
            return self.current_glucose
            
        # Weighted average giving more importance to recent readings
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # Weights for the last 5 readings
        weighted_sum = sum(w * reading['glucose_level'] for w, reading in zip(weights, recent_readings[-5:]))
        total_weight = sum(weights)
        
        # Include momentum from recent change
        momentum_factor = self.last_glucose_change * 0.7
        
        prediction = (weighted_sum / total_weight) + momentum_factor
        return max(40, min(400, prediction))  # Keep within reasonable bounds

    def get_active_carbs(self) -> float:
        """Calculate active carbs on board with more realistic modeling"""
        current_time = datetime.now(timezone.utc)
        active_carbs = 0
        
        for meal in self.meal_events:
            time_since_meal = (current_time - meal.timestamp).total_seconds() / 60
            if time_since_meal < meal.duration_minutes:
                # More realistic carb absorption curve
                if time_since_meal < 60:
                    # First hour: rapid absorption
                    remaining_ratio = 0.7 * (1 - time_since_meal / 60)
                else:
                    # Subsequent hours: slower absorption
                    remaining_ratio = 0.3 * (meal.duration_minutes - time_since_meal) / (meal.duration_minutes - 60)
                
                active_carbs += meal.carbs * remaining_ratio
                
        return round(active_carbs, 1)

    def get_active_insulin(self) -> float:
        """Calculate insulin on board with more realistic modeling"""
        current_time = datetime.now(timezone.utc)
        active_insulin = 0
        
        for dose in self.insulin_doses:
            time_since_dose = (current_time - dose.timestamp).total_seconds() / 60
            if time_since_dose < dose.duration_minutes:
                if dose.insulin_type == "rapid":
                    # More realistic insulin activity curve
                    if time_since_dose < 20:
                        activity = 0.1
                    elif time_since_dose < 75:
                        activity = 0.1 + 0.7 * ((time_since_dose - 20) / 55)
                    elif time_since_dose < 180:
                        activity = 0.8 - 0.3 * ((time_since_dose - 75) / 105)
                    else:
                        activity = 0.5 * (dose.duration_minutes - time_since_dose) / (dose.duration_minutes - 180)
                    
                    active_insulin += dose.units * activity * dose.effectiveness
                    
        return round(active_insulin, 2)

    def update_physiological_state(self):
        """Update various physiological parameters with more realistic patterns"""
        # Stress level varies throughout day with circadian pattern
        current_hour = datetime.now().hour
        base_stress = 1.0 + 0.3 * math.sin((current_hour - 14) * math.pi / 12)  # Higher in afternoon
        
        # Random stress events (less frequent)
        if random.random() < 0.005:  # 0.5% chance per reading
            self.current_stress = random.uniform(1.2, 1.8)
        else:
            # Gradual return to baseline
            self.current_stress = 0.9 * self.current_stress + 0.1 * base_stress
            
        # Hydration decreases gradually, increases with meals and drinks
        self.hydration = max(50, self.hydration - 0.05)  # Slower dehydration
        
        # Hydration increases with meals
        if self.last_meal_time and (datetime.now(timezone.utc) - self.last_meal_time).total_seconds() < 3600:
            self.hydration = min(100, self.hydration + 0.2)

    def generate_reading(self) -> GlucoseReading:
        """Generate comprehensive glucose reading with enhanced realism"""
        previous_glucose = self.current_glucose
        
        # Simulate daily schedule and events
        self.simulate_daily_schedule()
        self.update_physiological_state()
        
        # Calculate realistic glucose effects
        meal_insulin_effect, active_insulin = self.calculate_realistic_glucose_effects()
        circadian_effect = self.calculate_circadian_effects()
        
        # Stress and other factors
        stress_effect = (self.current_stress - 1.0) * 12  # Reduced effect for realism
        
        # Random physiological noise - reduced for more stable readings
        base_noise = random.gauss(0, 1.5)
        
        # Patient-type specific effects
        if self.sensor_config.patient_type == 'type1':
            # More variable, depends heavily on insulin
            variability_multiplier = 1.1  # Reduced variability
        elif self.sensor_config.patient_type == 'type2':
            # Less variable, some insulin resistance
            variability_multiplier = 0.9
            circadian_effect *= 1.2  # More dawn phenomenon
        else:
            variability_multiplier = 1.0
        
        # Combine all effects
        total_change = (
            base_noise + 
            meal_insulin_effect + 
            circadian_effect + 
            stress_effect
        ) * variability_multiplier * self.patient_profile.glucose_variability
        
        # Apply change with momentum for more realistic gradual changes
        new_glucose = (self.current_glucose * self.glucose_momentum) + ((self.current_glucose + total_change) * (1 - self.glucose_momentum))
        
        # Track rate of change for trend calculation
        self.last_glucose_change = new_glucose - self.current_glucose
        
        # Physiological bounds
        new_glucose = max(40, min(400, new_glucose))
        
        # Add sensor noise with slight calibration drift over time
        sensor_age_days = self.calculate_sensor_age()
        calibration_factor = 1.0 + self.calibration_drift * (sensor_age_days / 14)  # Drift increases over 14 days
        
        sensor_noise = random.gauss(0, 2) * (1 - self.sensor_config.sensor_accuracy)
        measured_glucose = new_glucose * calibration_factor + sensor_noise
        measured_glucose = max(40, min(400, measured_glucose))
        
        self.current_glucose = new_glucose
        
        # Store in history
        reading_data = {
            'glucose_level': measured_glucose,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.glucose_history.append(reading_data)
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.glucose_history = [
            r for r in self.glucose_history 
            if datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')) >= cutoff_time
        ]
        
        # Calculate dashboard metrics
        time_in_range_1h = self.calculate_time_in_range(1)
        time_in_range_24h = self.calculate_time_in_range(24)
        estimated_hba1c = self.estimate_hba1c()
        cv = self.calculate_glucose_variability()
        prediction_30min = self.predict_glucose_30min()
        
        # Determine trend and alert
        trend = self.determine_trend(previous_glucose, measured_glucose)
        alert_level = self.determine_alert_level(measured_glucose)
        
        # Get meal status
        meal_status = "none"
        if self.meal_events:
            last_meal = self.meal_events[-1]
            time_since_last_meal = (datetime.now(timezone.utc) - last_meal.timestamp).total_seconds() / 60
            if time_since_last_meal < 180:
                meal_status = f"{last_meal.meal_type.value}_{int(time_since_last_meal)}min_ago"
        
        # Calculate sleep quality (more realistic)
        sleep_quality = 85.0
        if self.activity_state == ActivityType.SLEEP and self.sleep_start:
            sleep_duration = (datetime.now() - self.sleep_start).total_seconds() / 3600
            # Sleep quality follows a curve: improves then decreases if too long
            if sleep_duration < 7:
                sleep_quality = 60 + sleep_duration * 8  # Improves with sleep duration
            else:
                sleep_quality = 116 - sleep_duration * 4  # Decreases after 7 hours
        
        # Create comprehensive reading
        reading = GlucoseReading(
            device_id=self.sensor_config.device_id,
            patient_id=self.sensor_config.patient_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            glucose_level=round(measured_glucose, 1),
            glucose_trend=trend.value,
            alert_level=alert_level.value,
            battery_level=self.calculate_battery_level(),
            signal_strength=random.randint(85, 100),  # Higher signal strength
            calibration_status="ok" if self.calculate_sensor_age() < 10 else "needs_calibration",
            sensor_age_days=self.calculate_sensor_age(),
            temperature=round(random.uniform(96.8, 98.6), 1),  # More normal range
            sequence_number=self.sequence_number,
            
            # Enhanced dashboard features
            time_in_range_1h=round(time_in_range_1h, 1),
            time_in_range_24h=round(time_in_range_24h, 1),
            estimated_hba1c=estimated_hba1c,
            glucose_variability_cv=round(cv, 1),
            predicted_glucose_30min=round(prediction_30min, 1),
            insulin_on_board=self.get_active_insulin(),
            carbs_on_board=self.get_active_carbs(),
            current_activity=self.activity_state.value,
            meal_status=meal_status,
            stress_level=round(self.current_stress, 2),
            sleep_quality=round(sleep_quality, 1),
            hydration_level=round(self.hydration, 1)
        )
        
        self.sequence_number += 1
        return reading

    def determine_trend(self, previous_glucose: float, current_glucose: float) -> GlucoseTrend:
        """Determine glucose trend based on rate of change with hysteresis"""
        rate_of_change = current_glucose - previous_glucose
        
        # Add hysteresis to prevent rapid flipping between states
        if abs(rate_of_change) < 0.8:
            return GlucoseTrend.STEADY
        elif rate_of_change >= 2.5:
            return GlucoseTrend.RISING_RAPIDLY
        elif rate_of_change >= 0.8:
            return GlucoseTrend.RISING
        elif rate_of_change <= -2.5:
            return GlucoseTrend.FALLING_RAPIDLY
        else:
            return GlucoseTrend.FALLING

    def determine_alert_level(self, glucose: float) -> AlertLevel:
        """Determine alert level based on glucose value with hysteresis"""
        if glucose < 54:
            return AlertLevel.CRITICAL_LOW
        elif glucose < 70:
            return AlertLevel.LOW
        elif glucose > 250:
            return AlertLevel.CRITICAL_HIGH
        elif glucose > 180:
            return AlertLevel.HIGH
        else:
            return AlertLevel.NORMAL

    def calculate_battery_level(self) -> int:
        """Calculate realistic battery drain with non-linear characteristics"""
        days_running = (datetime.now(timezone.utc) - self.sensor_start_time).total_seconds() / 86400
        
        # Non-linear battery drain: faster at beginning and end
        if days_running < 3:
            # Initial faster drain
            battery_used = days_running * self.battery_drain_rate * 1.2
        elif days_running > 12:
            # Faster drain as battery ages
            battery_used = days_running * self.battery_drain_rate * 1.3
        else:
            # Normal drain
            battery_used = days_running * self.battery_drain_rate
            
        current_battery = max(1, int(self.initial_battery - battery_used))
        return current_battery

    def calculate_sensor_age(self) -> int:
        """Calculate sensor age in days"""
        return (datetime.now(timezone.utc) - self.sensor_start_time).days

    def connect(self) -> bool:
        """Enhanced connection with retry logic"""
        if self.connection_attempts >= self.comm_config.max_retries:
            self.logger.error(f"Max connection attempts ({self.comm_config.max_retries}) exceeded")
            return False
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.comm_config.timeout)
            self.socket.connect((self.comm_config.host, self.comm_config.port))
            
            self.logger.info(f"Connected to MiNiFi agent at {self.comm_config.host}:{self.comm_config.port}")
            self.connection_attempts = 0
            return True
            
        except (socket.error, ConnectionRefusedError, TimeoutError) as e:
            self.connection_attempts += 1
            self.logger.error(f"Connection attempt {self.connection_attempts} failed: {e}")
            return False

    def send_data(self, data: Dict[str, Any]) -> bool:
        """Enhanced data sending with validation"""
        try:
            if not isinstance(data, dict) or 'glucose_level' not in data:
                self.logger.error("Invalid data format")
                return False
                
            message = json.dumps(data, default=str) + '\n'
            self.socket.sendall(message.encode('utf-8'))
            self.logger.debug(f"Sent: {message.strip()}")
            return True
            
        except (socket.error, BrokenPipeError, json.JSONEncodeError) as e:
            self.logger.error(f"Send failed: {e}")
            return False

    def send_heartbeat(self):
        """Send periodic heartbeat to maintain connection"""
        if time.time() - self.last_heartbeat > self.comm_config.heartbeat_interval:
            heartbeat = {
                "type": "heartbeat",
                "device_id": self.sensor_config.device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "online",
                "battery_level": self.calculate_battery_level(),
                "sensor_age_days": self.calculate_sensor_age()
            }
            
            if self.socket and self.send_data(heartbeat):
                self.last_heartbeat = time.time()
                self.logger.debug("Heartbeat sent")

    def run(self):
        """Enhanced main application loop"""
        self.running = True
        interval = self.sensor_config.update_interval_seconds
        
        self.logger.info(f"Starting Enhanced CGM Sensor with Dashboard Features")
        self.logger.info(f"Device: {self.sensor_config.device_id}")
        self.logger.info(f"Patient: {self.sensor_config.patient_id} ({self.sensor_config.patient_type})")
        self.logger.info(f"Update interval: {interval} seconds")
        self.logger.info(f"Target glucose range: {self.patient_profile.target_glucose_range[0]}-{self.patient_profile.target_glucose_range[1]} mg/dL")
        self.logger.info("Press Ctrl+C to stop.")
        
        if not self.connect():
            self.logger.warning("Initial connection failed. Will retry with data sends.")
        
        try:
            while self.running:
                reading = self.generate_reading()
                
                if not self.socket:
                    if not self.connect():
                        time.sleep(self.comm_config.retry_delay)
                        continue
                
                reading_dict = asdict(reading)
                if self.send_data(reading_dict):
                    # Enhanced logging with dashboard metrics
                    if reading.alert_level in ['critical_low', 'critical_high']:
                        self.logger.warning(f" CRITICAL ALERT - Glucose: {reading.glucose_level} mg/dL | "
                                      f"Trend: {reading.glucose_trend} | TIR(24h): {reading.time_in_range_24h}% | "
                                      f"IOB: {reading.insulin_on_board}U | COB: {reading.carbs_on_board}g")
                    elif reading.alert_level in ['low', 'high']:
                        self.logger.warning(f"  ALERT - Glucose: {reading.glucose_level} mg/dL | "
                                      f"Trend: {reading.glucose_trend} | Activity: {reading.current_activity}")
                    else:
                        if self.sequence_number % 12 == 0:  # Every minute for 5-second intervals
                            self.logger.info(f" Glucose: {reading.glucose_level} mg/dL | "
                                       f"Trend: {reading.glucose_trend} | "
                                       f"TIR(24h): {reading.time_in_range_24h}% | "
                                       f"Est. A1C: {reading.estimated_hba1c}% | "
                                       f"CV: {reading.glucose_variability_cv}% | "
                                       f"Prediction(30min): {reading.predicted_glucose_30min} mg/dL")
                        else:
                            self.logger.info(f"Glucose: {reading.glucose_level} mg/dL | {reading.glucose_trend}")
                else:
                    if self.socket:
                        self.socket.close()
                        self.socket = None
                
                self.send_heartbeat()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal...")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources and close connections"""
        self.logger.info("Cleaning up resources...")
        self.running = False
        
        if self.socket:
            try:
                # Send shutdown message
                shutdown_msg = {
                    "type": "shutdown",
                    "device_id": self.sensor_config.device_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "normal_shutdown"
                }
                self.socket.sendall(json.dumps(shutdown_msg).encode('utf-8'))
            except:
                pass
            
            try:
                self.socket.close()
            except:
                pass
                
        self.logger.info("Enhanced CGM Sensor shutdown complete")

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Mock CGM Sensor with Dashboard Features')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')
    parser.add_argument('--interval', '-i', type=int, help='Update interval in seconds')
    parser.add_argument('--host', type=str, help='MiNiFi agent host')
    parser.add_argument('--port', type=int, help='MiNiFi agent port')
    parser.add_argument('--device-id', type=str, help='Device ID')
    parser.add_argument('--patient-id', type=str, help='Patient ID')
    parser.add_argument('--patient-type', type=str, choices=['type1', 'type2', 'prediabetes', 'gestational'], 
                       help='Patient diabetes type')
    
    args = parser.parse_args()
    
    # Create sensor instance
    sensor = EnhancedMockCGMSensor(args.config)
    
    # Override config with command line arguments if provided
    if args.interval:
        sensor.sensor_config.update_interval_seconds = args.interval
    if args.host:
        sensor.comm_config.host = args.host
    if args.port:
        sensor.comm_config.port = args.port
    if args.device_id:
        sensor.sensor_config.device_id = args.device_id
    if args.patient_id:
        sensor.sensor_config.patient_id = args.patient_id
    if args.patient_type:
        sensor.sensor_config.patient_type = args.patient_type
    
    try:
        sensor.run()
    except Exception as e:
        sensor.logger.error(f"Fatal error: {e}", exc_info=True)
        sensor.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()