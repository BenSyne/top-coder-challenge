#!/usr/bin/env python3
"""
Test and calibrate the reimbursement calculator
"""

import json
import pandas as pd
from calculate_reimbursement import ReimbursementCalculator

def test_specific_cases():
    """Test specific cases to understand the calculation pattern"""
    calculator = ReimbursementCalculator()
    
    test_cases = [
        # (days, miles, receipts, expected)
        (3, 93, 1.42, 364.51),
        (1, 55, 3.6, 126.06),
        (1, 47, 17.97, 128.91),
        (5, 130, 306.90, 574.10),
        (5, 173, 1337.90, 1443.96),
    ]
    
    print("Testing specific cases:")
    print("-" * 80)
    print(f"{'Days':>5} {'Miles':>6} {'Receipts':>10} {'Expected':>10} {'Calculated':>12} {'Error':>10}")
    print("-" * 80)
    
    for days, miles, receipts, expected in test_cases:
        calculated = calculator.calculate(days, miles, receipts)
        error = calculated - expected
        print(f"{days:5d} {miles:6d} {receipts:10.2f} {expected:10.2f} {calculated:12.2f} {error:10.2f}")
    
def analyze_components():
    """Break down calculation components for sample cases"""
    calculator = ReimbursementCalculator()
    
    # Test case: 3 days, 93 miles, $1.42 receipts (expected: $364.51)
    days, miles, receipts = 3, 93, 1.42
    
    print("\nComponent breakdown for: 3 days, 93 miles, $1.42 receipts")
    print("-" * 60)
    
    base = calculator.calculate_base_per_diem(days)
    mileage = calculator.calculate_mileage_reimbursement(miles, days)
    receipt_reimb = calculator.calculate_receipt_reimbursement(receipts, days)
    efficiency = calculator.calculate_efficiency_bonus(miles, days)
    
    total_before_adjustments = base + mileage + receipt_reimb + efficiency
    
    print(f"Base per diem: ${base:.2f}")
    print(f"Mileage reimbursement: ${mileage:.2f}")
    print(f"Receipt reimbursement: ${receipt_reimb:.2f}")
    print(f"Efficiency bonus: ${efficiency:.2f}")
    print(f"Total before adjustments: ${total_before_adjustments:.2f}")
    
    # Apply adjustments
    adjusted = calculator.apply_trip_category_adjustments(total_before_adjustments, days, miles, receipts)
    final = calculator.apply_bugs_and_quirks(adjusted, receipts)
    
    print(f"After category adjustments: ${adjusted:.2f}")
    print(f"After bugs/quirks: ${final:.2f}")
    print(f"Expected: $364.51")
    print(f"Error: ${final - 364.51:.2f}")

def analyze_averages():
    """Analyze average expected reimbursements by category"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame([{
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'],
        'receipts': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    } for case in data])
    
    print("\n\nAverage expected reimbursements by trip length:")
    print("-" * 40)
    for days in range(1, 15):
        subset = df[df['days'] == days]
        if len(subset) > 0:
            avg = subset['expected'].mean()
            print(f"{days} days: ${avg:.2f} (n={len(subset)})")

if __name__ == "__main__":
    test_specific_cases()
    analyze_components()
    analyze_averages() 