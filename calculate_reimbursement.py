#!/usr/bin/env python3
"""
Black Box Reimbursement System - Calculator Implementation
Phase 2: Polynomial model based on advanced analysis
"""

import sys
import math


class ReimbursementCalculator:
    """Calculates travel reimbursement using polynomial regression model"""
    
    def __init__(self):
        # Polynomial coefficients from analysis
        # Model: days, miles, receipts, days^2, days*miles, days*receipts, miles^2, miles*receipts, receipts^2
        self.intercept = 64.09
        
        # Linear terms
        self.coef_days = 88.172302
        self.coef_miles = 0.406955
        self.coef_receipts = 1.211677
        
        # Quadratic terms
        self.coef_days_squared = -2.590275
        self.coef_miles_squared = 0.000035
        self.coef_receipts_squared = -0.000279
        
        # Interaction terms
        self.coef_days_miles = 0.014510
        self.coef_days_receipts = -0.008909
        self.coef_miles_receipts = -0.000114
        
        # Special adjustments based on patterns
        self.rounding_bug_factor = 0.41  # Receipts ending in .49/.99
        self.efficiency_bonus = 50.0  # For 180-220 miles/day
        
    def calculate_polynomial_features(self, days, miles, receipts):
        """Calculate polynomial and interaction features"""
        features = {
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'days_squared': days * days,
            'miles_squared': miles * miles,
            'receipts_squared': receipts * receipts,
            'days_miles': days * miles,
            'days_receipts': days * receipts,
            'miles_receipts': miles * receipts
        }
        return features
    
    def calculate_base_amount(self, days, miles, receipts):
        """Calculate base reimbursement using polynomial model"""
        features = self.calculate_polynomial_features(days, miles, receipts)
        
        amount = self.intercept
        amount += self.coef_days * features['days']
        amount += self.coef_miles * features['miles']
        amount += self.coef_receipts * features['receipts']
        amount += self.coef_days_squared * features['days_squared']
        amount += self.coef_miles_squared * features['miles_squared']
        amount += self.coef_receipts_squared * features['receipts_squared']
        amount += self.coef_days_miles * features['days_miles']
        amount += self.coef_days_receipts * features['days_receipts']
        amount += self.coef_miles_receipts * features['miles_receipts']
        
        return amount
    
    def apply_special_patterns(self, amount, days, miles, receipts):
        """Apply special patterns discovered in analysis"""
        miles_per_day = miles / days if days > 0 else miles
        
        # Efficiency bonus for optimal miles/day
        if 180 <= miles_per_day <= 220:
            amount += self.efficiency_bonus
        
        # Rounding bug (significant reduction)
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
            amount *= self.rounding_bug_factor
        
        # Additional adjustments for edge cases
        # Very short trips with low miles need special handling
        if days == 1 and miles < 100 and receipts < 50:
            # These tend to be overestimated by polynomial
            amount *= 0.65
        
        # Long trips with very high mileage need boost
        if days >= 7 and miles > 1000:
            # These are systematically underestimated
            amount *= 1.35
        
        # 5-day trips seem to have a slight penalty
        if days == 5:
            amount *= 0.95
        
        return amount
    
    def calculate(self, days, miles, receipts):
        """Main calculation method"""
        # Calculate base polynomial amount
        amount = self.calculate_base_amount(days, miles, receipts)
        
        # Apply special patterns and adjustments
        amount = self.apply_special_patterns(amount, days, miles, receipts)
        
        # Ensure non-negative and round to 2 decimal places
        return max(0.0, round(amount, 2))


def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) != 4:
        print("Usage: calculate_reimbursement.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        miles = int(miles)
        receipts = float(sys.argv[3])
        
        calculator = ReimbursementCalculator()
        result = calculator.calculate(days, miles, receipts)
        
        # Output only the result
        print(f"{result:.2f}")
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 