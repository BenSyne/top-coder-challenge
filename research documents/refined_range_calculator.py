#!/usr/bin/env python3
"""
Refined range calculator - remove conflicts and optimize further
"""

import sys

class RefinedRangeCalculator:
    """Calculator with refined range-specific adjustments"""
    
    def __init__(self):
        # Base coefficients
        self.intercept = 266.71
        self.coef_days = 50.05
        self.coef_miles = 0.4456
        self.coef_receipts = 0.3829
        self.rounding_bug_factor = 0.457
        
        # Refined range multipliers
        self.range_multipliers = {
            (1, 2): 1.05,    # Short trips
            (3, 4): 1.10,    # Medium trips  
            (5, 7): 1.15,    # Week trips (no more conflicting 5-day penalty!)
            (8, 11): 1.02,   # Mid-long trips (slight boost)
            (12, 14): 0.95,  # Very long trips (slight reduction)
            (15, 30): 1.05   # Extended trips
        }
        
    def get_range_multiplier(self, days):
        """Get the appropriate multiplier for trip length"""
        for (min_days, max_days), multiplier in self.range_multipliers.items():
            if min_days <= days <= max_days:
                return multiplier
        return 1.00
        
    def calculate(self, days, miles, receipts):
        """Calculate with refined range optimizations"""
        
        # Check for rounding bug
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
            # Keep only non-conflicting adjustments
            # REMOVED: 5-day penalty (conflicts with range multiplier)
            
            # Keep high-mileage long trip reduction
            if days >= 12 and miles >= 1000:
                amount *= 0.85
            
            # Keep minimal trip boost  
            if days <= 2 and miles < 100 and receipts < 50:
                amount *= 1.15
            
            # Keep high-value trip boosts
            if 7 <= days <= 8 and miles >= 1000 and receipts >= 1000:
                amount *= 1.25
            
            if 13 <= days <= 14 and miles >= 1000 and receipts >= 1000:
                amount *= 1.20
            
            # Keep efficiency bonus
            miles_per_day = miles / days if days > 0 else miles
            if 180 <= miles_per_day <= 220:
                amount += 30.0
        
        # Apply range-specific multiplier
        if not has_rounding_bug:
            range_multiplier = self.get_range_multiplier(days)
            amount *= range_multiplier
        
        return max(0.0, round(amount, 2))

def test_refined_calculator():
    """Test the refined calculator"""
    
    import json
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calculator = RefinedRangeCalculator()
    
    total_error = 0
    exact_matches = 0
    
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = abs(predicted - expected)
        total_error += error
        
        if error <= 0.01:
            exact_matches += 1
    
    avg_error = total_error / len(data)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print("🔧 REFINED RANGE CALCULATOR:")
    print("=" * 50)
    print(f"Exact matches: {exact_matches}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Score: {score:.0f}")
    
    prev_score = 14616
    improvement = prev_score - score
    print(f"Additional improvement: {improvement:.0f} points")
    
    total_improvement = 16840 - score
    print(f"Total improvement: {total_improvement:.0f} points")

def main():
    """Command-line interface"""
    if len(sys.argv) != 4:
        print("Usage: refined_range_calculator.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        miles = int(miles)
        receipts = float(sys.argv[3])
        
        calculator = RefinedRangeCalculator()
        result = calculator.calculate(days, miles, receipts)
        
        print(f"{result:.2f}")
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        test_refined_calculator()
    else:
        main() 