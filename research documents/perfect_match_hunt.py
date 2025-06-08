#!/usr/bin/env python3
"""
Hunt for exact matches by testing tiny adjustments
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from calculate_reimbursement import ReimbursementCalculator

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return data

def test_tiny_adjustments():
    """Test tiny adjustments to see if we can get exact matches"""
    data = load_data()
    calculator = ReimbursementCalculator()
    
    # Get the closest cases
    close_cases = []
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = abs(predicted - expected)
        
        if error < 10:  # Focus on cases within $10
            close_cases.append({
                'index': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': predicted,
                'error': error
            })
    
    # Sort by error
    close_cases.sort(key=lambda x: x['error'])
    
    print("HUNTING FOR EXACT MATCHES")
    print("=" * 60)
    print("Testing tiny adjustments on closest cases...\n")
    
    # Test various tiny adjustments
    adjustments = [
        ("Round to nearest $0.05", lambda x: round(x * 20) / 20),
        ("Round to nearest $0.10", lambda x: round(x * 10) / 10),
        ("Round to nearest $0.25", lambda x: round(x * 4) / 4),
        ("Round to nearest $1.00", lambda x: round(x)),
        ("Ceiling", lambda x: __import__('math').ceil(x)),
        ("Floor", lambda x: __import__('math').floor(x)),
        ("Add $0.01", lambda x: x + 0.01),
        ("Subtract $0.01", lambda x: x - 0.01),
        ("Add $0.02", lambda x: x + 0.02),
        ("Subtract $0.02", lambda x: x - 0.02),
        ("Multiply by 1.001", lambda x: x * 1.001),
        ("Multiply by 0.999", lambda x: x * 0.999),
    ]
    
    best_improvement = None
    best_exact_matches = 0
    
    for adj_name, adj_func in adjustments:
        exact_matches = 0
        total_error = 0
        
        for case in close_cases[:50]:  # Test on closest 50 cases
            adjusted = adj_func(case['predicted'])
            if abs(adjusted - case['expected']) < 0.01:
                exact_matches += 1
            total_error += abs(adjusted - case['expected'])
        
        avg_error = total_error / min(50, len(close_cases))
        
        if exact_matches > best_exact_matches:
            best_exact_matches = exact_matches
            best_improvement = (adj_name, adj_func, avg_error)
        
        print(f"{adj_name:20}: {exact_matches:2d} exact matches, avg error ${avg_error:.3f}")
    
    if best_improvement:
        print(f"\n🎯 BEST ADJUSTMENT: {best_improvement[0]}")
        print(f"   Exact matches: {best_exact_matches}")
        print(f"   Average error: ${best_improvement[2]:.3f}")
    
    return best_improvement

def analyze_specific_patterns():
    """Look for specific patterns in the closest cases"""
    data = load_data()
    calculator = ReimbursementCalculator()
    
    print("\n\nANALYZING SPECIFIC PATTERNS")
    print("=" * 60)
    
    # Look at the closest case
    days, miles, receipts = 1, 420, 2273.60
    expected = 1220.35
    predicted = calculator.calculate(days, miles, receipts)
    
    print(f"CLOSEST CASE: {days}d, {miles}mi, ${receipts:.2f}")
    print(f"Expected: ${expected:.2f}")
    print(f"Predicted: ${predicted:.2f}")
    print(f"Error: ${predicted - expected:.4f}")
    
    # Calculate the base amount step by step
    base = 266.71 + 50.05 * days + 0.4456 * miles + 0.3829 * receipts
    print(f"Base calculation: ${base:.4f}")
    
    # Check what adjustments were applied
    print("\nStep-by-step breakdown:")
    print(f"1. Base: 266.71 + 50.05×{days} + 0.4456×{miles} + 0.3829×{receipts:.2f} = ${base:.2f}")
    
    # Check for rounding bug
    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
        print(f"2. Rounding bug applied: ${base:.2f} × 0.457 = ${base * 0.457:.2f}")
    else:
        print("2. No rounding bug")
    
    # What would make it exact?
    needed_adjustment = expected - predicted
    print(f"\nTo be exact, we need: {needed_adjustment:+.4f}")
    print(f"That's a factor of: {expected/predicted:.6f}")

if __name__ == "__main__":
    best_adj = test_tiny_adjustments()
    analyze_specific_patterns() 