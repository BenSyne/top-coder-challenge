#!/usr/bin/env python3
"""
Strategic error analysis - focus on biggest wins for score improvement
"""

import json
import math
from collections import defaultdict, Counter

def analyze_worst_cases():
    """Deep dive into our worst-performing cases to find systematic patterns"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Our current best calculator
    def current_calculation(days, miles, receipts):
        # Apply caps
        capped_receipts = receipts
        if receipts > 1800:
            capped_receipts = 1800 + (receipts - 1800) * 0.15
        
        capped_miles = miles
        if miles > 800:
            capped_miles = 800 + (miles - 800) * 0.25
        
        # Base calculation
        amount = 266.71 + 50.05 * days + 0.4456 * capped_miles + 0.3829 * capped_receipts
        
        # Check for rounding bug
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
            amount *= 0.457
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
        
        return round(amount, 2)
    
    # Calculate errors for all cases
    cases_with_errors = []
    over_predictions = 0
    under_predictions = 0
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = current_calculation(days, miles, receipts)
        error = predicted - expected  # Signed error
        abs_error = abs(error)
        
        if error > 0:
            over_predictions += 1
        else:
            under_predictions += 1
        
        cases_with_errors.append({
            'case': i,
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'expected': expected,
            'predicted': predicted,
            'error': error,
            'abs_error': abs_error,
            'ratio': expected / predicted if predicted > 0 else 0,
            'miles_per_day': miles / days if days > 0 else miles,
            'receipts_per_day': receipts / days if days > 0 else receipts
        })
    
    print("SYSTEMATIC OVER/UNDER PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"Over-predictions: {over_predictions} ({over_predictions/len(data)*100:.1f}%)")
    print(f"Under-predictions: {under_predictions} ({under_predictions/len(data)*100:.1f}%)")
    
    # Sort by absolute error to find worst cases
    worst_cases = sorted(cases_with_errors, key=lambda x: x['abs_error'], reverse=True)
    
    print(f"\nTOP 20 WORST CASES (Highest absolute errors):")
    print("-" * 60)
    for i, case in enumerate(worst_cases[:20]):
        direction = "OVER" if case['error'] > 0 else "UNDER"
        print(f"{i+1:2d}. Case {case['case']:3d}: {case['days']:2d}d, {case['miles']:4.0f}mi, ${case['receipts']:7.2f}")
        print(f"     Expected: ${case['expected']:7.2f}, Got: ${case['predicted']:7.2f}")
        print(f"     Error: ${case['error']:+7.2f} ({direction}), Ratio: {case['ratio']:.4f}")
    
    return worst_cases

def find_correction_patterns(worst_cases):
    """Look for patterns that could give us a systematic correction"""
    
    print(f"\n\nCORRECTION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Group by ratio ranges to find systematic adjustments
    ratio_groups = defaultdict(list)
    for case in worst_cases[:100]:  # Top 100 worst cases
        if case['ratio'] > 0:
            ratio_bucket = round(case['ratio'], 1)  # Round to nearest 0.1
            ratio_groups[ratio_bucket].append(case)
    
    print("Ratio distribution in worst 100 cases:")
    for ratio in sorted(ratio_groups.keys()):
        cases_in_group = len(ratio_groups[ratio])
        if cases_in_group >= 3:  # Only show significant groups
            avg_days = sum(c['days'] for c in ratio_groups[ratio]) / cases_in_group
            avg_miles = sum(c['miles'] for c in ratio_groups[ratio]) / cases_in_group
            avg_receipts = sum(c['receipts'] for c in ratio_groups[ratio]) / cases_in_group
            print(f"  Ratio {ratio:.1f}: {cases_in_group:2d} cases (avg: {avg_days:.1f}d, {avg_miles:.0f}mi, ${avg_receipts:.0f})")
    
    # Look for day-specific patterns
    print(f"\nDay-specific error patterns:")
    day_errors = defaultdict(list)
    for case in worst_cases[:50]:
        day_errors[case['days']].append(case['ratio'])
    
    for days in sorted(day_errors.keys()):
        ratios = day_errors[days]
        if len(ratios) >= 2:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"  {days:2d} days: {len(ratios):2d} cases, avg ratio: {avg_ratio:.4f}")

def test_global_adjustment_factor():
    """Test if a simple global adjustment factor could improve our score"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n\nGLOBAL ADJUSTMENT FACTOR TESTING")
    print("=" * 60)
    
    # Test different global multipliers
    factors = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02]
    
    best_score = float('inf')
    best_factor = 1.0
    
    for factor in factors:
        total_error = 0
        exact_matches = 0
        
        for case in data:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            # Apply our formula then global adjustment
            predicted_base = calculate_base_amount(days, miles, receipts)
            predicted = predicted_base * factor
            predicted = round(predicted, 2)
            predicted = max(0.0, predicted)
            
            error = abs(predicted - expected)
            total_error += error
            
            if error <= 0.01:
                exact_matches += 1
        
        avg_error = total_error / len(data)
        score = avg_error * 100 + (1000 - exact_matches) * 0.1
        
        print(f"Factor {factor:.2f}: Avg error ${avg_error:6.2f}, {exact_matches} exact, Score: {score:7.1f}")
        
        if score < best_score:
            best_score = score
            best_factor = factor
    
    print(f"\nBest factor: {best_factor:.2f} with score: {best_score:.1f}")
    improvement = 16840 - best_score
    print(f"Potential improvement: {improvement:.1f} points")

def calculate_base_amount(days, miles, receipts):
    """Our current base calculation"""
    # Apply caps
    capped_receipts = receipts
    if receipts > 1800:
        capped_receipts = 1800 + (receipts - 1800) * 0.15
    
    capped_miles = miles
    if miles > 800:
        capped_miles = 800 + (miles - 800) * 0.25
    
    # Base calculation
    amount = 266.71 + 50.05 * days + 0.4456 * capped_miles + 0.3829 * capped_receipts
    
    # Check for rounding bug
    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
        amount *= 0.457
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
    
    return amount

def test_range_specific_adjustments():
    """Test if specific ranges need different adjustment factors"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n\nRANGE-SPECIFIC ADJUSTMENT TESTING")
    print("=" * 60)
    
    # Test different adjustments for different day ranges
    day_ranges = [
        (1, 2, "Short trips"),
        (3, 4, "Medium trips"), 
        (5, 7, "Week trips"),
        (8, 14, "Long trips"),
        (15, 30, "Extended trips")
    ]
    
    for min_days, max_days, label in day_ranges:
        relevant_cases = [case for case in data 
                         if min_days <= case['input']['trip_duration_days'] <= max_days]
        
        if len(relevant_cases) < 10:  # Skip ranges with too few cases
            continue
        
        print(f"\n{label} ({min_days}-{max_days} days): {len(relevant_cases)} cases")
        
        # Test different factors for this range
        best_error = float('inf')
        best_factor = 1.0
        
        for factor in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
            total_error = 0
            
            for case in relevant_cases:
                days = case['input']['trip_duration_days']
                miles = case['input']['miles_traveled']
                receipts = case['input']['total_receipts_amount']
                expected = case['expected_output']
                
                predicted_base = calculate_base_amount(days, miles, receipts)
                predicted = predicted_base * factor
                predicted = round(predicted, 2)
                predicted = max(0.0, predicted)
                
                error = abs(predicted - expected)
                total_error += error
            
            avg_error = total_error / len(relevant_cases)
            
            if avg_error < best_error:
                best_error = avg_error
                best_factor = factor
        
        print(f"  Best factor: {best_factor:.2f}, Avg error: ${best_error:.2f}")

if __name__ == "__main__":
    worst_cases = analyze_worst_cases()
    find_correction_patterns(worst_cases)
    test_global_adjustment_factor()
    test_range_specific_adjustments() 