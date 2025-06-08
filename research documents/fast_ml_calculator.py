#!/usr/bin/env python3
"""
Fast ML Calculator - Pre-trains models once for speed
Target: Beat score 8,800 while maintaining fast performance
"""

import sys
import json
import pickle
import os

# Try importing ML libraries
try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available", file=sys.stderr)

MODEL_CACHE_FILE = "ml_models_cache.pkl"

class FastMLCalculator:
    """Fast ML calculator with pre-trained models"""
    
    def __init__(self):
        # Surgical model coefficients
        self.surgical_intercept = 266.71
        self.surgical_coef_days = 50.05
        self.surgical_coef_miles = 0.4456
        self.surgical_coef_receipts = 0.3829
        self.rounding_bug_factor = 0.457
        
        self.range_multipliers = {
            (1, 2): 1.06, (3, 4): 1.09, (5, 7): 1.14, 
            (8, 14): 1.01, (15, 30): 1.05
        }
        
        # ML components
        self.models = []
        self.scaler = None
        self.is_trained = False
        
        # Load or train models
        if ML_AVAILABLE:
            if os.path.exists(MODEL_CACHE_FILE):
                self._load_models()
            else:
                self._train_and_save_models()
    
    def _load_models(self):
        """Load pre-trained models from cache"""
        try:
            with open(MODEL_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                self.models = cache['models']
                self.scaler = cache['scaler']
                self.is_trained = True
                print("Loaded pre-trained models from cache", file=sys.stderr)
        except Exception as e:
            print(f"Error loading models: {e}", file=sys.stderr)
            self._train_and_save_models()
    
    def _train_and_save_models(self):
        """Train models once and save to cache"""
        if not ML_AVAILABLE:
            return
            
        try:
            print("Training ML models (one-time process)...", file=sys.stderr)
            
            # Load training data
            with open('public_cases.json', 'r') as f:
                data = json.load(f)
            
            # Prepare features and residuals
            X = []
            residuals = []
            
            for case in data:
                days = case['input']['trip_duration_days']
                miles = case['input']['miles_traveled']
                receipts = case['input']['total_receipts_amount']
                expected = case['expected_output']
                
                features = self.create_features(days, miles, receipts)
                surgical_pred = self.surgical_calculate(days, miles, receipts)
                residual = expected - surgical_pred
                
                X.append(features)
                residuals.append(residual)
            
            X = np.array(X)
            residuals = np.array(residuals)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train enhanced ensemble with more models
            self.models = [
                ('rf1', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
                ('rf2', RandomForestRegressor(n_estimators=150, max_depth=20, random_state=123)),
                ('et', ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42)),
                ('gb1', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)),
                ('gb2', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=123)),
                ('mlp1', MLPRegressor(hidden_layer_sizes=(150, 100, 50), max_iter=1000, random_state=42)),
                ('mlp2', MLPRegressor(hidden_layer_sizes=(200, 100), max_iter=800, random_state=123))
            ]
            
            # Train all models
            for name, model in self.models:
                print(f"Training {name}...", file=sys.stderr)
                model.fit(X_scaled, residuals)
            
            self.is_trained = True
            
            # Save to cache
            cache = {
                'models': self.models,
                'scaler': self.scaler
            }
            with open(MODEL_CACHE_FILE, 'wb') as f:
                pickle.dump(cache, f)
            
            print("Models trained and saved to cache!", file=sys.stderr)
            
        except Exception as e:
            print(f"Error training models: {e}", file=sys.stderr)
            self.is_trained = False
    
    def get_range_multiplier(self, days):
        """Get range multiplier"""
        for (min_days, max_days), multiplier in self.range_multipliers.items():
            if min_days <= days <= max_days:
                return multiplier
        return 1.00
    
    def surgical_calculate(self, days, miles, receipts):
        """Surgical baseline calculation"""
        receipt_str = f"{receipts:.2f}"
        has_rounding_bug = receipt_str.endswith('.49') or receipt_str.endswith('.99')
        
        # Apply caps
        capped_receipts = receipts
        if receipts > 1800:
            capped_receipts = 1800 + (receipts - 1800) * 0.15
        
        capped_miles = miles
        if miles > 800:
            capped_miles = 800 + (miles - 800) * 0.25
        
        # Base calculation
        amount = (self.surgical_intercept + 
                 self.surgical_coef_days * days + 
                 self.surgical_coef_miles * capped_miles + 
                 self.surgical_coef_receipts * capped_receipts)
        
        if has_rounding_bug:
            amount *= self.rounding_bug_factor
        else:
            # All adjustments
            if days >= 12 and miles >= 1000:
                amount *= 0.85
            if days <= 2 and miles < 100 and receipts < 50:
                amount *= 1.15
            if 7 <= days <= 8 and miles >= 1000 and receipts >= 1000:
                amount *= 1.25
            if 13 <= days <= 14 and miles >= 1000 and receipts >= 1000:
                amount *= 1.20
            
            miles_per_day = miles / days if days > 0 else miles
            if 180 <= miles_per_day <= 220:
                amount += 30.0
            
            # Range multipliers
            amount *= self.get_range_multiplier(days)
            
            # Surgical fixes
            if days == 7 and receipts > 2000:
                amount *= 0.85
            if 8 <= days <= 14 and 900 <= receipts <= 1500 and miles < 1200:
                amount *= 1.15
            if days >= 12 and miles >= 1000 and receipts < 1200:
                amount *= 1.10
            if 8 <= days <= 11 and miles >= 1000 and receipts >= 1000:
                amount *= 0.95
        
        return max(0.0, round(amount, 2))
    
    def create_features(self, days, miles, receipts):
        """Create enhanced feature set"""
        features = []
        
        # Basic features
        features.extend([days, miles, receipts])
        
        # Ratios
        miles_per_day = miles / days if days > 0 else miles
        receipts_per_day = receipts / days if days > 0 else receipts
        receipts_per_mile = receipts / miles if miles > 0 else receipts
        features.extend([miles_per_day, receipts_per_day, receipts_per_mile])
        
        # Polynomial features
        features.extend([days**2, miles**2, receipts**2, days**3, miles**3, receipts**3])
        features.extend([days*miles, days*receipts, miles*receipts])
        features.extend([days*miles*receipts])
        
        # Log features
        features.extend([
            np.log(max(1, days)), 
            np.log(max(1, miles)), 
            np.log(max(1, receipts)),
            np.log(max(1, miles_per_day)),
            np.log(max(1, receipts_per_day))
        ])
        
        # Categorical (trip length)
        features.extend([
            1 if 1 <= days <= 2 else 0,
            1 if 3 <= days <= 4 else 0,
            1 if 5 <= days <= 7 else 0,
            1 if 8 <= days <= 14 else 0,
            1 if days >= 15 else 0
        ])
        
        # Categorical (miles)
        features.extend([
            1 if miles < 100 else 0,
            1 if 100 <= miles < 500 else 0,
            1 if 500 <= miles < 1000 else 0,
            1 if 1000 <= miles < 2000 else 0,
            1 if miles >= 2000 else 0
        ])
        
        # Categorical (receipts)
        features.extend([
            1 if receipts < 500 else 0,
            1 if 500 <= receipts < 1000 else 0,
            1 if 1000 <= receipts < 1500 else 0,
            1 if 1500 <= receipts < 2000 else 0,
            1 if receipts >= 2000 else 0
        ])
        
        # Special patterns
        receipt_str = f"{receipts:.2f}"
        features.append(1 if receipt_str.endswith('.49') or receipt_str.endswith('.99') else 0)
        features.append(1 if 180 <= miles_per_day <= 220 else 0)
        features.append(1 if receipts > miles else 0)
        features.append(1 if days == 7 else 0)
        features.append(1 if days >= 12 and miles >= 1000 else 0)
        
        return np.array(features)
    
    def calculate(self, days, miles, receipts):
        """Fast ML-enhanced calculation"""
        # Get surgical baseline
        surgical_pred = self.surgical_calculate(days, miles, receipts)
        
        # If ML not available or not trained, return surgical
        if not ML_AVAILABLE or not self.is_trained:
            return surgical_pred
        
        try:
            # Get features and scale
            features = self.create_features(days, miles, receipts).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from all models
            residual_predictions = []
            for name, model in self.models:
                pred = model.predict(features_scaled)[0]
                residual_predictions.append(pred)
            
            # Weighted ensemble (give more weight to better models)
            weights = [1.2, 1.0, 1.1, 1.2, 1.0, 0.9, 0.8]  # Tuned weights
            weighted_residual = np.average(residual_predictions, weights=weights)
            
            # Combine surgical + ML residual
            final_pred = surgical_pred + weighted_residual
            
            return max(0.0, round(final_pred, 2))
            
        except Exception as e:
            # Fallback to surgical
            return surgical_pred

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Fast ML reimbursement calculation"""
    # Use singleton pattern for efficiency
    if not hasattr(calculate_reimbursement, 'calculator'):
        calculate_reimbursement.calculator = FastMLCalculator()
    
    return calculate_reimbursement.calculator.calculate(
        trip_duration_days, miles_traveled, total_receipts_amount
    )

def main():
    """Command-line interface"""
    if len(sys.argv) != 4:
        print("Usage: calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = int(float(sys.argv[2]))
        receipts = float(sys.argv[3])
        
        result = calculate_reimbursement(days, miles, receipts)
        print(f"{result:.2f}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 