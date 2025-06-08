#!/usr/bin/env python3
"""
Fine-tune coefficients to maximize exact matches
"""

import json
import math
from itertools import product

def test_coefficient_variations():
    """Test small variations in coefficients"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Current best coefficients
    base_intercept = 266.71
    base_days = 50.05
    base_miles = 0.4456
    base_receipts = 0.3829
    
    # Test small variations
    intercept_range = [base_intercept + x for x in [-1.0, -0.5, 0, 0.5, 1.0]]
    days_range = [base_days + x for x in [-0.2, -0.1, 0, 0.1, 0.2]]
    miles_range = [base_miles + x for x in [-0.01, -0.005, 0, 0.005, 0.01]]
    receipts_range = [base_receipts + x for x in [-0.01, -0.005, 0, 0.005, 0.01]]
    
    best_exact_matches = 1  # We already have 1
    best_coeffs = None
    best_score = float('inf')
    
    print("TESTING COEFFICIENT VARIATIONS")
    print("=" * 50)
    print("Testing variations around our best coefficients...")
    
    # Test a subset to avoid too many combinations
    test_count = 0
    max_tests = 500  # Limit to reasonable number
    
    for intercept in intercept_range[::2]:  # Skip some values
        for days in days_range[::2]:
            for miles in miles_range:
                for receipts in receipts_range:
                    if test_count >= max_tests:
                        break
                    
                    exact_matches = 0
                    total_error = 0
                    
                    for case in data:
                        case_days = case['input']['trip_duration_days']
                        case_miles = case['input']['miles_traveled']
                        case_receipts = case['input']['total_receipts_amount']
                        expected = case['expected_output']
                        
                        # Apply our formula with new coefficients
                        predicted = calculate_with_coeffs(
                            case_days, case_miles, case_receipts,
                            intercept, days, miles, receipts
                        )
                        
                        error = abs(predicted - expected)
                        total_error += error
                        
                        if error <= 0.01:
                            exact_matches += 1
                    
                    test_count += 1
                    
                    if exact_matches > best_exact_matches or \
                       (exact_matches == best_exact_matches and total_error < best_score):
                        best_exact_matches = exact_matches
                        best_coeffs = (intercept, days, miles, receipts)
                        best_score = total_error
                        
                        print(f"New best: {exact_matches} exact matches, score: {total_error:.0f}")
                        print(f"  Coeffs: {intercept:.2f}, {days:.3f}, {miles:.4f}, {receipts:.4f}")
                    
                    if test_count % 100 == 0:
                        print(f"Tested {test_count} combinations...")
    
    print(f"\nFinal best: {best_exact_matches} exact matches")
    if best_coeffs:
        print(f"Best coefficients: {best_coeffs}")
    else:
        print("No improvement found")

def calculate_with_coeffs(days, miles, receipts, intercept, coef_days, coef_miles, coef_receipts):
    """Calculate reimbursement with given coefficients"""
    
    # Apply caps
    capped_receipts = receipts
    if receipts > 1800:
        capped_receipts = 1800 + (receipts - 1800) * 0.15
    
    capped_miles = miles
    if miles > 800:
        capped_miles = 800 + (miles - 800) * 0.25
    
    # Base calculation with new coefficients
    amount = intercept + coef_days * days + coef_miles * capped_miles + coef_receipts * capped_receipts
    
    # Check for rounding bug
    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
        amount *= 0.457
    else:
        # Apply adjustments
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
    
    # Nickel rounding (our best rounding method)
    result = round(amount * 20) / 20
    
    return max(0.0, result)

def test_alternative_formulas():
    """Test completely different formula approaches"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("\n\nTESTING ALTERNATIVE FORMULAS")
    print("=" * 50)
    
    # Test some completely different approaches
    formulas = [
        ("Power law", lambda d, m, r: 100 * (d ** 0.9) + 0.4 * (m ** 0.95) + 0.35 * (r ** 0.9)),
        ("Log components", lambda d, m, r: 50 * d + 100 * math.log(m + 1) + 150 * math.log(r + 1)),
        ("Square root", lambda d, m, r: 80 * d + 10 * math.sqrt(m) + 20 * math.sqrt(r)),
        ("Piecewise linear", lambda d, m, r: min(150 * d, 100 * d + 0.5 * m + 0.4 * r)),
    ]
    
    for name, formula in formulas:
        exact_matches = 0
        total_error = 0
        
        for case in data:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            try:
                base = formula(days, miles, receipts)
                
                # Apply rounding bug if needed
                receipt_str = f"{receipts:.2f}"
                if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
                    base *= 0.457
                
                # Nickel rounding
                predicted = round(base * 20) / 20
                predicted = max(0.0, predicted)
                
                error = abs(predicted - expected)
                total_error += error
                
                if error <= 0.01:
                    exact_matches += 1
            except:
                total_error += 1000  # Penalty for formula failure
        
        avg_error = total_error / len(data)
        print(f"{name:20}: {exact_matches} exact matches, avg error: ${avg_error:.2f}")

if __name__ == "__main__":
    test_coefficient_variations()
    test_alternative_formulas() 