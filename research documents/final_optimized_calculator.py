#!/usr/bin/env python3
"""
Final optimized calculator with 2 exact matches
"""

import sys
import math

class FinalOptimizedCalculator:
    """Final calculator with optimized coefficients"""
    
    def __init__(self):
        # Optimized coefficients that give us 2 exact matches!
        self.intercept = 267.71
        self.coef_days = 49.85
        self.coef_miles = 0.4406
        self.coef_receipts = 0.3779
        
        # Rounding bug factor
        self.rounding_bug_factor = 0.457
        
    def calculate(self, days, miles, receipts):
        """Main calculation method with optimized coefficients"""
        
        # Check for rounding bug first
        receipt_str = f"{receipts:.2f}"
        has_rounding_bug = receipt_str.endswith('.49') or receipt_str.endswith('.99')
        
        # Apply caps
        capped_receipts = receipts
        if receipts > 1800:
            capped_receipts = 1800 + (receipts - 1800) * 0.15
        
        capped_miles = miles
        if miles > 800:
            capped_miles = 800 + (miles - 800) * 0.25
        
        # Base calculation with optimized coefficients
        amount = (self.intercept + 
                 self.coef_days * days + 
                 self.coef_miles * capped_miles + 
                 self.coef_receipts * capped_receipts)
        
        # Apply rounding bug
        if has_rounding_bug:
            amount *= self.rounding_bug_factor
        else:
            # Other adjustments only if no rounding bug
            
            # 5-day trips need reduction
            if days == 5:
                amount *= 0.92
            
            # Very long trips with high mileage need reduction
            if days >= 12 and miles >= 1000:
                amount *= 0.85
            
            # Short trips with minimal inputs need boost
            if days <= 2 and miles < 100 and receipts < 50:
                amount *= 1.15
            
            # Mid-length trips with high mileage need boost
            if 7 <= days <= 8 and miles >= 1000 and receipts >= 1000:
                amount *= 1.25
            
            # 13-14 day trips with high values need boost
            if 13 <= days <= 14 and miles >= 1000 and receipts >= 1000:
                amount *= 1.20
            
            # Efficiency bonus
            miles_per_day = miles / days if days > 0 else miles
            if 180 <= miles_per_day <= 220:
                amount += 30.0
        
        # CRITICAL: Use nickel rounding ($0.05) for exact matches
        result = round(amount * 20) / 20
        
        # Ensure non-negative
        return max(0.0, result)

def test_final_performance():
    """Test the final optimized calculator"""
    
    import json
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calculator = FinalOptimizedCalculator()
    
    exact_matches = 0
    close_matches = 0
    total_error = 0
    exact_cases = []
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = abs(predicted - expected)
        total_error += error
        
        if error <= 0.01:
            exact_matches += 1
            exact_cases.append(i)
            print(f"🎯 EXACT MATCH #{exact_matches}: Case {i}")
            print(f"   {days}d, {miles}mi, ${receipts:.2f} → Expected: ${expected:.2f}, Got: ${predicted:.2f}")
        
        if error <= 1.00:
            close_matches += 1
    
    avg_error = total_error / len(data)
    score = total_error
    
    print(f"\n🏆 FINAL OPTIMIZED RESULTS:")
    print(f"Exact matches: {exact_matches}")
    print(f"Close matches: {close_matches}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Score: {score:.0f}")
    
    improvement = 16840 - score
    print(f"Improvement over original: {improvement:.0f} points!")
    
    return exact_matches, score

def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) != 4:
        print("Usage: final_optimized_calculator.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        miles = int(miles)
        receipts = float(sys.argv[3])
        
        calculator = FinalOptimizedCalculator()
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
    if len(sys.argv) == 1:
        # Run test if no command line args
        test_final_performance()
    else:
        main() 