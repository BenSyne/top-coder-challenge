#!/usr/bin/env python3
"""
Test very simple formulas to see if we're overcomplicating
"""

import json
import numpy as np

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    return data

def test_formula(data, formula_func, name):
    """Test a formula and return average error"""
    total_error = 0
    exact_matches = 0
    close_matches = 0
    errors = []
    
    for case in data:
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = formula_func(days, miles, receipts)
        error = abs(predicted - expected)
        errors.append(error)
        total_error += error
        
        if error <= 0.01:
            exact_matches += 1
        if error <= 1.00:
            close_matches += 1
    
    avg_error = total_error / len(data)
    max_error = max(errors)
    
    return {
        'name': name,
        'avg_error': avg_error,
        'max_error': max_error,
        'exact_matches': exact_matches,
        'close_matches': close_matches
    }

def main():
    data = load_data()
    
    # Test various simple formulas
    formulas = [
        # Basic linear combinations
        ("100*d + 0.50*m + 0.50*r", lambda d, m, r: 100*d + 0.50*m + 0.50*r),
        ("80*d + 0.45*m + 0.45*r", lambda d, m, r: 80*d + 0.45*m + 0.45*r),
        ("60*d + 0.55*m + 0.35*r", lambda d, m, r: 60*d + 0.55*m + 0.35*r),
        ("50*d + 0.60*m + 0.40*r", lambda d, m, r: 50*d + 0.60*m + 0.40*r),
        
        # With intercepts
        ("200 + 40*d + 0.50*m + 0.35*r", lambda d, m, r: 200 + 40*d + 0.50*m + 0.35*r),
        ("150 + 50*d + 0.45*m + 0.40*r", lambda d, m, r: 150 + 50*d + 0.45*m + 0.40*r),
        
        # Capped formulas
        ("100*d + 0.5*m + 0.4*min(r,1500)", lambda d, m, r: 100*d + 0.5*m + 0.4*min(r, 1500)),
        ("80*d + 0.5*min(m,800) + 0.4*r", lambda d, m, r: 80*d + 0.5*min(m, 800) + 0.4*r),
        
        # Combined caps
        ("75*d + 0.5*min(m,1000) + 0.4*min(r,1800)", 
         lambda d, m, r: 75*d + 0.5*min(m, 1000) + 0.4*min(r, 1800)),
        
        # Percentage based
        ("0.1*(d*750 + m + r)", lambda d, m, r: 0.1*(d*750 + m + r)),
        ("0.15*(d*500 + m + r)", lambda d, m, r: 0.15*(d*500 + m + r)),
        
        # Square root formulas (diminishing returns)
        ("100*d + 5*sqrt(m) + 10*sqrt(r)", lambda d, m, r: 100*d + 5*np.sqrt(m) + 10*np.sqrt(r)),
        ("80*d + 8*sqrt(m) + 15*sqrt(r)", lambda d, m, r: 80*d + 8*np.sqrt(m) + 15*np.sqrt(r)),
        
        # Log formulas
        ("50*d + 50*log(m+1) + 100*log(r+1)", lambda d, m, r: 50*d + 50*np.log(m+1) + 100*np.log(r+1)),
        
        # Min/max formulas
        ("max(100*d, 0.5*(m+r))", lambda d, m, r: max(100*d, 0.5*(m+r))),
        ("min(200*d, 100*d + 0.4*(m+r))", lambda d, m, r: min(200*d, 100*d + 0.4*(m+r))),
    ]
    
    results = []
    for name, func in formulas:
        result = test_formula(data, func, name)
        results.append(result)
    
    # Sort by average error
    results.sort(key=lambda x: x['avg_error'])
    
    print("SIMPLE FORMULA TEST RESULTS")
    print("=" * 80)
    print(f"{'Formula':<40} {'Avg Error':>10} {'Max Error':>10} {'Exact':>6} {'Close':>6}")
    print("-" * 80)
    
    for r in results[:10]:  # Top 10
        print(f"{r['name']:<40} ${r['avg_error']:>9.2f} ${r['max_error']:>9.2f} "
              f"{r['exact_matches']:>6d} {r['close_matches']:>6d}")
    
    # Test the best formula on specific problem cases
    print("\n\nTesting best formula on problem cases:")
    best_formula = formulas[results[0]['name'] == formulas[0][0]][0][1]
    
    problem_cases = [
        (7, 1006, 1181.33, 2279.82),
        (14, 1020, 1201.75, 2337.73),
        (13, 1167, 1074.36, 2197.33)
    ]
    
    for days, miles, receipts, expected in problem_cases:
        predicted = results[0]['name']
        formula_func = None
        for name, func in formulas:
            if name == results[0]['name']:
                formula_func = func
                break
        
        if formula_func:
            pred = formula_func(days, miles, receipts)
            error = pred - expected
            print(f"{days}d, {miles}mi, ${receipts:.2f}: "
                  f"Expected ${expected:.2f}, Got ${pred:.2f}, Error ${error:.2f}")

if __name__ == "__main__":
    main() 