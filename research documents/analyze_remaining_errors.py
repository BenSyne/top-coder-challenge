#!/usr/bin/env python3
"""
Analyze remaining errors in Fast ML Calculator to find improvement opportunities
"""

import json
import numpy as np
from fast_ml_calculator import FastMLCalculator

def analyze_errors():
    """Deep dive into remaining errors"""
    
    print("🔍 ANALYZING REMAINING ERRORS IN FAST ML MODEL")
    print("=" * 60)
    
    # Load test data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Initialize calculator
    calculator = FastMLCalculator()
    
    # Collect all errors
    errors = []
    large_errors = []
    by_category = {
        'rounding_bug': [],
        'short_trip': [],
        'medium_trip': [],
        'week_trip': [],
        'long_trip': [],
        'extended_trip': [],
        'high_miles': [],
        'high_receipts': [],
        'low_values': []
    }
    
    print("Calculating errors...")
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = predicted - expected  # Signed error
        abs_error = abs(error)
        
        error_info = {
            'index': i,
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'expected': expected,
            'predicted': predicted,
            'error': error,
            'abs_error': abs_error,
            'pct_error': abs_error / expected * 100 if expected > 0 else 0
        }
        
        errors.append(error_info)
        
        # Categorize errors
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
            by_category['rounding_bug'].append(error_info)
        
        if days <= 2:
            by_category['short_trip'].append(error_info)
        elif days <= 4:
            by_category['medium_trip'].append(error_info)
        elif days <= 7:
            by_category['week_trip'].append(error_info)
        elif days <= 14:
            by_category['long_trip'].append(error_info)
        else:
            by_category['extended_trip'].append(error_info)
        
        if miles >= 1000:
            by_category['high_miles'].append(error_info)
        if receipts >= 2000:
            by_category['high_receipts'].append(error_info)
        if days <= 3 and miles <= 100 and receipts <= 100:
            by_category['low_values'].append(error_info)
        
        if abs_error > 100:
            large_errors.append(error_info)
    
    # Sort by absolute error
    errors_sorted = sorted(errors, key=lambda x: x['abs_error'], reverse=True)
    
    # Statistics
    all_errors = [e['error'] for e in errors]
    abs_errors = [e['abs_error'] for e in errors]
    
    print(f"\n📊 ERROR STATISTICS")
    print("-" * 40)
    print(f"Total cases: {len(errors)}")
    print(f"Average error: ${np.mean(abs_errors):.2f}")
    print(f"Median error: ${np.median(abs_errors):.2f}")
    print(f"Std deviation: ${np.std(abs_errors):.2f}")
    print(f"Max error: ${max(abs_errors):.2f}")
    print(f"Cases > $100 error: {len(large_errors)} ({len(large_errors)/len(errors)*100:.1f}%)")
    
    # Directional bias
    over_predictions = sum(1 for e in all_errors if e > 0)
    under_predictions = sum(1 for e in all_errors if e < 0)
    print(f"\nOver-predictions: {over_predictions} ({over_predictions/len(errors)*100:.1f}%)")
    print(f"Under-predictions: {under_predictions} ({under_predictions/len(errors)*100:.1f}%)")
    print(f"Mean signed error: ${np.mean(all_errors):.2f}")
    
    # Category analysis
    print(f"\n📂 ERROR BY CATEGORY")
    print("-" * 40)
    for category, cases in by_category.items():
        if cases:
            cat_errors = [c['abs_error'] for c in cases]
            print(f"{category:15}: {len(cases):4d} cases, ${np.mean(cat_errors):6.2f} avg error")
    
    # Worst cases
    print(f"\n❌ TOP 20 WORST PREDICTIONS")
    print("-" * 60)
    print("Idx  Days Miles  Receipts   Expected  Predicted    Error   %Err")
    print("-" * 60)
    for e in errors_sorted[:20]:
        print(f"{e['index']:3d}  {e['days']:4d} {e['miles']:5.0f} {e['receipts']:9.2f} "
              f"{e['expected']:9.2f} {e['predicted']:9.2f} {e['error']:8.2f} {e['pct_error']:5.1f}%")
    
    # Pattern detection in worst cases
    print(f"\n🔍 PATTERNS IN WORST CASES (>$100 error)")
    print("-" * 40)
    
    if large_errors:
        large_days = [e['days'] for e in large_errors]
        large_miles = [e['miles'] for e in large_errors]
        large_receipts = [e['receipts'] for e in large_errors]
        
        print(f"Days: mean={np.mean(large_days):.1f}, median={np.median(large_days):.0f}")
        print(f"Miles: mean={np.mean(large_miles):.0f}, median={np.median(large_miles):.0f}")
        print(f"Receipts: mean=${np.mean(large_receipts):.0f}, median=${np.median(large_receipts):.0f}")
        
        # Check for common patterns
        rounding_bug_count = sum(1 for e in large_errors 
                                if f"{e['receipts']:.2f}".endswith(('.49', '.99')))
        print(f"Rounding bug cases: {rounding_bug_count} ({rounding_bug_count/len(large_errors)*100:.0f}%)")
        
        high_value_count = sum(1 for e in large_errors 
                              if e['receipts'] > 2000 or e['miles'] > 1500)
        print(f"High value cases: {high_value_count} ({high_value_count/len(large_errors)*100:.0f}%)")
    
    # Error distribution
    print(f"\n📈 ERROR DISTRIBUTION")
    print("-" * 40)
    ranges = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 500), (500, 1000)]
    for low, high in ranges:
        count = sum(1 for e in abs_errors if low <= e < high)
        pct = count / len(abs_errors) * 100
        print(f"${low:3d}-${high:3d}: {count:4d} cases ({pct:5.1f}%)")
    count = sum(1 for e in abs_errors if e >= 1000)
    pct = count / len(abs_errors) * 100
    print(f"$1000+  : {count:4d} cases ({pct:5.1f}%)")
    
    return errors_sorted, by_category

def find_improvement_opportunities(errors_sorted, by_category):
    """Identify specific areas for improvement"""
    
    print(f"\n💡 IMPROVEMENT OPPORTUNITIES")
    print("=" * 60)
    
    # Check if certain trip lengths are consistently off
    print("\n1. Trip Length Bias:")
    for days in range(1, 31):
        day_errors = [e for e in errors_sorted if e['days'] == days]
        if len(day_errors) >= 5:  # Only if we have enough samples
            avg_signed = np.mean([e['error'] for e in day_errors])
            avg_abs = np.mean([e['abs_error'] for e in day_errors])
            if abs(avg_signed) > 20:  # Significant bias
                print(f"   Day {days:2d}: {len(day_errors):3d} cases, "
                      f"bias=${avg_signed:6.2f}, avg error=${avg_abs:6.2f}")
    
    # Check mile ranges
    print("\n2. Mileage Range Issues:")
    mile_ranges = [(0, 100), (100, 300), (300, 500), (500, 800), (800, 1200), (1200, 2000), (2000, 5000)]
    for low, high in mile_ranges:
        range_errors = [e for e in errors_sorted if low <= e['miles'] < high]
        if range_errors:
            avg_signed = np.mean([e['error'] for e in range_errors])
            avg_abs = np.mean([e['abs_error'] for e in range_errors])
            if abs(avg_signed) > 15:
                print(f"   {low:4d}-{high:4d}mi: {len(range_errors):3d} cases, "
                      f"bias=${avg_signed:6.2f}, avg error=${avg_abs:6.2f}")
    
    # Check receipt ranges
    print("\n3. Receipt Range Issues:")
    receipt_ranges = [(0, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000), (3000, 5000)]
    for low, high in receipt_ranges:
        range_errors = [e for e in errors_sorted if low <= e['receipts'] < high]
        if range_errors:
            avg_signed = np.mean([e['error'] for e in range_errors])
            avg_abs = np.mean([e['abs_error'] for e in range_errors])
            if abs(avg_signed) > 15:
                print(f"   ${low:4d}-${high:4d}: {len(range_errors):3d} cases, "
                      f"bias=${avg_signed:6.2f}, avg error=${avg_abs:6.2f}")
    
    # Interaction patterns
    print("\n4. Interaction Patterns:")
    # High miles + low receipts
    pattern1 = [e for e in errors_sorted if e['miles'] > 1000 and e['receipts'] < 500]
    if pattern1:
        avg_error = np.mean([e['abs_error'] for e in pattern1])
        print(f"   High miles + low receipts: {len(pattern1)} cases, ${avg_error:.2f} avg error")
    
    # Short trip + high receipts
    pattern2 = [e for e in errors_sorted if e['days'] <= 3 and e['receipts'] > 1500]
    if pattern2:
        avg_error = np.mean([e['abs_error'] for e in pattern2])
        print(f"   Short trip + high receipts: {len(pattern2)} cases, ${avg_error:.2f} avg error")
    
    # Week trips with specific receipt ranges
    pattern3 = [e for e in errors_sorted if e['days'] == 7 and 1000 <= e['receipts'] <= 2000]
    if pattern3:
        avg_error = np.mean([e['abs_error'] for e in pattern3])
        avg_signed = np.mean([e['error'] for e in pattern3])
        print(f"   7-day trips ($1k-$2k receipts): {len(pattern3)} cases, ${avg_error:.2f} avg error, ${avg_signed:.2f} bias")

def main():
    """Run error analysis"""
    errors_sorted, by_category = analyze_errors()
    find_improvement_opportunities(errors_sorted, by_category)
    
    print("\n🎯 RECOMMENDATIONS:")
    print("=" * 60)
    print("1. Add bias correction terms for specific day counts")
    print("2. Enhance feature engineering for edge cases") 
    print("3. Consider separate models for rounding bug vs normal cases")
    print("4. Add more interaction features (miles×receipts/days)")
    print("5. Implement meta-learning layer to correct systematic biases")

if __name__ == "__main__":
    main() 