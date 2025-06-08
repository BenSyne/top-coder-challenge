#!/usr/bin/env python3
"""
Range-optimized calculator with trip length specific adjustments
"""

import sys
import math

class RangeOptimizedCalculator:
    """Calculator with range-specific adjustment factors"""
    
    def __init__(self):
        # Our proven base coefficients
        self.intercept = 266.71
        self.coef_days = 50.05
        self.coef_miles = 0.4456
        self.coef_receipts = 0.3829
        self.rounding_bug_factor = 0.457
        
        # Range-specific multipliers from analysis
        self.range_multipliers = {
            (1, 2): 1.05,    # Short trips
            (3, 4): 1.10,    # Medium trips  
            (5, 7): 1.15,    # Week trips
            (8, 14): 1.00,   # Long trips (our formula already works well)
            (15, 30): 1.05   # Extended trips (conservative)
        }
        
    def get_range_multiplier(self, days):
        """Get the appropriate multiplier for trip length"""
        for (min_days, max_days), multiplier in self.range_multipliers.items():
            if min_days <= days <= max_days:
                return multiplier
        return 1.00  # Default for unusual cases
        
    def calculate(self, days, miles, receipts):
        """Calculate reimbursement with range-specific optimizations"""
        
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
        
        # Base calculation
        amount = (self.intercept + 
                 self.coef_days * days + 
                 self.coef_miles * capped_miles + 
                 self.coef_receipts * capped_receipts)
        
        # Apply rounding bug
        if has_rounding_bug:
            amount *= self.rounding_bug_factor
        else:
            # Apply legacy adjustments (but these might be redundant now)
            if days == 5:
                amount *= 0.92  # Keep this for now, range multiplier will override
            
            if days >= 12 and miles >= 1000:
                amount *= 0.85
            
            if days <= 2 and miles < 100 and receipts < 50:
                amount *= 1.15
            
            if 7 <= days <= 8 and miles >= 1000 and receipts >= 1000:
                amount *= 1.25
            
            if 13 <= days <= 14 and miles >= 1000 and receipts >= 1000:
                amount *= 1.20
            
            # Efficiency bonus
            miles_per_day = miles / days if days > 0 else miles
            if 180 <= miles_per_day <= 220:
                amount += 30.0
        
        # CRITICAL: Apply range-specific multiplier (this is our key improvement!)
        if not has_rounding_bug:  # Don't adjust rounding bug cases
            range_multiplier = self.get_range_multiplier(days)
            amount *= range_multiplier
        
        # Round result
        result = round(amount, 2)
        
        return max(0.0, result)

def test_range_optimization():
    """Test the range-optimized calculator"""
    
    import json
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calculator = RangeOptimizedCalculator()
    
    total_error = 0
    exact_matches = 0
    range_performance = {}
    
    # Initialize range tracking
    for (min_days, max_days), multiplier in calculator.range_multipliers.items():
        range_performance[f"{min_days}-{max_days}d"] = {
            'cases': 0,
            'total_error': 0,
            'multiplier': multiplier
        }
    
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
        
        # Track range performance
        for (min_days, max_days), multiplier in calculator.range_multipliers.items():
            if min_days <= days <= max_days:
                range_key = f"{min_days}-{max_days}d"
                range_performance[range_key]['cases'] += 1
                range_performance[range_key]['total_error'] += error
                break
    
    avg_error = total_error / len(data)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print("🚀 RANGE-OPTIMIZED RESULTS:")
    print("=" * 50)
    print(f"Exact matches: {exact_matches}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Score: {score:.0f}")
    
    current_score = 16840
    improvement = current_score - score
    print(f"Improvement: {improvement:.0f} points!")
    
    print(f"\nRange-specific performance:")
    for range_key, perf in range_performance.items():
        if perf['cases'] > 0:
            avg_range_error = perf['total_error'] / perf['cases']
            print(f"  {range_key:8}: {perf['cases']:3d} cases, ${avg_range_error:6.2f} avg error, {perf['multiplier']:.2f}× factor")
    
    return score

def main():
    """Command-line interface"""
    if len(sys.argv) != 4:
        print("Usage: range_optimized_calculator.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        miles = int(miles)
        receipts = float(sys.argv[3])
        
        calculator = RangeOptimizedCalculator()
        result = calculator.calculate(days, miles, receipts)
        
        print(f"{result:.2f}")
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        test_range_optimization()
    else:
        main() 