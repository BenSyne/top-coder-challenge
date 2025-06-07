#!/usr/bin/env python3
"""
Black Box Reimbursement System - Calculator Implementation
Best performing model with incremental improvements
"""

import sys
import math


class ReimbursementCalculator:
    """Calculates travel reimbursement using linear model with caps"""
    
    def __init__(self):
        # Linear regression coefficients (best performing)
        self.intercept = 266.71
        self.coef_days = 50.05
        self.coef_miles = 0.4456
        self.coef_receipts = 0.3829
        
        # Rounding bug factor
        self.rounding_bug_factor = 0.457
        
    def calculate(self, days, miles, receipts):
        """Main calculation method"""
        
        # Check for rounding bug first
        receipt_str = f"{receipts:.2f}"
        has_rounding_bug = receipt_str.endswith('.49') or receipt_str.endswith('.99')
        
        # Apply caps based on analysis
        capped_receipts = receipts
        if receipts > 1800:
            # Receipts above $1800 processed at lower rate
            capped_receipts = 1800 + (receipts - 1800) * 0.15
        
        capped_miles = miles
        if miles > 800:
            # Miles above 800 get reduced rate
            capped_miles = 800 + (miles - 800) * 0.25
        
        # Base linear calculation with capped values
        amount = (self.intercept + 
                 self.coef_days * days + 
                 self.coef_miles * capped_miles + 
                 self.coef_receipts * capped_receipts)
        
        # Apply rounding bug
        if has_rounding_bug:
            amount *= self.rounding_bug_factor
        else:
            # Other adjustments only if no rounding bug
            
            # 5-day trips need reduction (they were over-predicted)
            if days == 5:
                amount *= 0.92
            
            # Very long trips with high mileage need reduction
            if days >= 12 and miles >= 1000:
                amount *= 0.85
            
            # Short trips with minimal inputs need boost
            if days <= 2 and miles < 100 and receipts < 50:
                amount *= 1.15
            
            # Mid-length trips (7-8 days) with high mileage need moderate boost
            # This was under-predicted in error analysis
            if 7 <= days <= 8 and miles >= 1000 and receipts >= 1000:
                amount *= 1.25
            
            # 13-14 day trips with high values also under-predicted
            if 13 <= days <= 14 and miles >= 1000 and receipts >= 1000:
                amount *= 1.20
            
            # Efficiency bonus
            miles_per_day = miles / days if days > 0 else miles
            if 180 <= miles_per_day <= 220:
                amount += 30.0
        
        # Round to 2 decimal places
        result = round(amount, 2)
        
        # Ensure non-negative
        return max(0.0, result)


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