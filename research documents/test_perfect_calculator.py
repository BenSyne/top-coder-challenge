#!/usr/bin/env python3
"""
Test calculator with rounding to nearest $0.05
"""

import sys
import math
import json

class TestReimbursementCalculator:
    """Test calculator with nickel rounding"""
    
    def __init__(self):
        # Same coefficients as our best model
        self.intercept = 266.71
        self.coef_days = 50.05
        self.coef_miles = 0.4456
        self.coef_receipts = 0.3829
        self.rounding_bug_factor = 0.457
        
    def calculate(self, days, miles, receipts):
        """Calculate with nickel rounding"""
        
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
            # Other adjustments
            if days == 5:
                amount *= 0.92
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
        
        # NEW: Round to nearest nickel ($0.05)
        result = round(amount * 20) / 20
        
        return max(0.0, result)

def test_nickel_rounding():
    """Test the nickel rounding approach"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calculator = TestReimbursementCalculator()
    
    exact_matches = 0
    close_matches = 0
    total_error = 0
    
    best_matches = []
    
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
            print(f"🎯 EXACT MATCH #{exact_matches}: Case {i}")
            print(f"   {days}d, {miles}mi, ${receipts:.2f} → Expected: ${expected:.2f}, Got: ${predicted:.2f}")
        
        if error <= 1.00:
            close_matches += 1
        
        if error < 5:  # Track very close matches
            best_matches.append({
                'case': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': predicted,
                'error': error
            })
    
    avg_error = total_error / len(data)
    score = total_error
    
    print(f"\nNICKEL ROUNDING RESULTS:")
    print(f"Exact matches: {exact_matches}")
    print(f"Close matches: {close_matches}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Score: {score:.0f}")
    
    if exact_matches > 0:
        print(f"\n🎉 SUCCESS! Found {exact_matches} exact matches!")
    
    # Show closest misses
    best_matches.sort(key=lambda x: x['error'])
    print(f"\nClosest misses:")
    for match in best_matches[:10]:
        if match['error'] > 0.01:  # Don't repeat exact matches
            print(f"  Case {match['case']}: {match['days']}d, {match['miles']}mi, ${match['receipts']:.2f}")
            print(f"    Expected: ${match['expected']:.2f}, Got: ${match['predicted']:.2f}, Error: ${match['error']:.3f}")

if __name__ == "__main__":
    test_nickel_rounding() 