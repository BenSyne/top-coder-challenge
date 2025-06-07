#!/usr/bin/env python3
"""
Analyze close matches to understand successful patterns
"""

import pandas as pd
import json
from calculate_reimbursement import ReimbursementCalculator

def analyze_close_matches():
    """Analyze the cases that are almost perfect matches"""
    
    # Load the error analysis results
    df = pd.read_csv('error_analysis_results.csv')
    
    # Get close matches (within $1)
    close_matches = df[df['abs_error'] <= 1.00].copy()
    
    print("CLOSE MATCHES ANALYSIS")
    print("=" * 60)
    print(f"Found {len(close_matches)} close matches\n")
    
    # Display each close match with full details
    for idx, row in close_matches.iterrows():
        print(f"Case {idx + 1}:")
        print(f"  Inputs: {row['days']} days, {row['miles']} miles, ${row['receipts']:.2f} receipts")
        print(f"  Expected: ${row['expected']:.2f}")
        print(f"  Predicted: ${row['predicted']:.2f}")
        print(f"  Error: ${row['error']:.2f}")
        print(f"  Miles/day: {row['miles_per_day']:.1f}")
        print(f"  Receipts/day: ${row['receipts_per_day']:.2f}")
        print(f"  Has rounding bug: {row['rounding_bug']}")
        
        # Calculate what the linear model would predict
        linear_pred = 64.09 + 88.17 * row['days'] + 0.407 * row['miles'] + 1.212 * row['receipts']
        print(f"  Pure polynomial would predict: ${linear_pred:.2f}")
        print(f"  Adjustment factor applied: {row['predicted'] / linear_pred:.3f}")
        print()
    
    # Look for common patterns
    print("\nPATTERNS IN CLOSE MATCHES:")
    print("-" * 40)
    print(f"Average days: {close_matches['days'].mean():.1f}")
    print(f"Average miles: {close_matches['miles'].mean():.0f}")
    print(f"Average receipts: ${close_matches['receipts'].mean():.2f}")
    print(f"Average miles/day: {close_matches['miles_per_day'].mean():.0f}")
    print(f"Average receipts/day: ${close_matches['receipts_per_day'].mean():.2f}")
    
    # Check trip types
    print("\nTrip type distribution:")
    print(f"  Short trips (1-3 days): {len(close_matches[close_matches['days'] <= 3])}")
    print(f"  Medium trips (4-7 days): {len(close_matches[(close_matches['days'] >= 4) & (close_matches['days'] <= 7)])}")
    print(f"  Long trips (8+ days): {len(close_matches[close_matches['days'] >= 8])}")
    
    return close_matches

def test_simple_formula():
    """Test if a simpler formula might work better"""
    
    print("\n\nTESTING SIMPLIFIED FORMULAS")
    print("=" * 60)
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Test different simple formulas
    formulas = [
        # (name, function)
        ("Linear only", lambda d, m, r: 100 * d + 0.45 * m + 0.45 * r),
        ("Days + Half inputs", lambda d, m, r: 100 * d + 0.5 * (m + r)),
        ("Weighted sum", lambda d, m, r: 75 * d + 0.55 * m + 0.35 * r),
        ("With intercept", lambda d, m, r: 150 + 45 * d + 0.52 * m + 0.38 * r),
    ]
    
    for name, formula in formulas:
        total_error = 0
        exact_matches = 0
        close_matches = 0
        
        for case in data[:100]:  # Test on first 100 cases
            days = case['input']['trip_duration_days']
            miles = int(case['input']['miles_traveled'])
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            predicted = formula(days, miles, receipts)
            error = abs(predicted - expected)
            total_error += error
            
            if error <= 0.01:
                exact_matches += 1
            if error <= 1.00:
                close_matches += 1
        
        avg_error = total_error / 100
        print(f"\n{name}:")
        print(f"  Average error: ${avg_error:.2f}")
        print(f"  Exact matches: {exact_matches}")
        print(f"  Close matches: {close_matches}")

def analyze_exact_formula_possibility():
    """Check if the expected values follow exact mathematical patterns"""
    
    print("\n\nEXACT FORMULA ANALYSIS")
    print("=" * 60)
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Check if outputs are always rounded to cents
    print("Checking decimal patterns in expected outputs:")
    decimal_counts = {}
    
    for case in data:
        expected = case['expected_output']
        decimal_part = f"{expected:.2f}".split('.')[-1]
        decimal_counts[decimal_part] = decimal_counts.get(decimal_part, 0) + 1
    
    # Show most common decimal endings
    sorted_decimals = sorted(decimal_counts.items(), key=lambda x: x[1], reverse=True)
    print("Most common decimal endings:")
    for decimal, count in sorted_decimals[:10]:
        print(f"  .{decimal}: {count} times")
    
    # Check for integer patterns
    print("\nChecking for integer patterns:")
    integer_outputs = [case for case in data if case['expected_output'] == int(case['expected_output'])]
    print(f"Outputs that are exact integers: {len(integer_outputs)}")
    
    # Check for patterns in small trips
    print("\nAnalyzing 1-day trips with minimal inputs:")
    one_day_minimal = [case for case in data 
                       if case['input']['trip_duration_days'] == 1 
                       and case['input']['miles_traveled'] < 100
                       and case['input']['total_receipts_amount'] < 50]
    
    for case in one_day_minimal[:5]:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        print(f"  {days}d, {miles}mi, ${receipts:.2f} → ${expected:.2f}")

if __name__ == "__main__":
    close_matches = analyze_close_matches()
    test_simple_formula()
    analyze_exact_formula_possibility() 