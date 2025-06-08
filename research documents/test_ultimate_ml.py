#!/usr/bin/env python3
"""Test the ultimate ML calculator - pushing for near-zero error"""

import json
from ultimate_ml_calculator import UltimateMLCalculator
import time
import numpy as np

def test_score():
    """Test score on public cases"""
    
    print("🚀 Testing Ultimate ML Calculator - The Final Push!")
    print("=" * 70)
    print("Target: Score < 100 (99%+ accuracy)")
    print()
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Initialize calculator
    start_time = time.time()
    calculator = UltimateMLCalculator()
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.1f} seconds")
    
    total_error = 0
    exact_matches = 0
    near_perfect = 0
    improvements = 0
    
    # Track error distribution
    error_buckets = {
        'perfect': 0,    # < $0.01
        'excellent': 0,  # < $0.10
        'great': 0,      # < $0.50
        'good': 0,       # < $1.00
        'ok': 0,         # < $5.00
        'poor': 0        # >= $5.00
    }
    
    print("\nProcessing test cases...")
    start_time = time.time()
    
    errors = []
    
    for i, case in enumerate(data):
        if i % 100 == 0:
            print(f"Progress: {i}/1000")
        
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = abs(predicted - expected)
        errors.append(error)
        total_error += error
        
        # Categorize error
        if error < 0.01:
            exact_matches += 1
            error_buckets['perfect'] += 1
        elif error < 0.10:
            error_buckets['excellent'] += 1
        elif error < 0.50:
            error_buckets['great'] += 1
        elif error < 1.00:
            error_buckets['good'] += 1
            near_perfect += 1
        elif error < 5.00:
            error_buckets['ok'] += 1
        else:
            error_buckets['poor'] += 1
        
        if error < 7.52:  # Beat Ultra ML average
            improvements += 1
    
    process_time = time.time() - start_time
    
    avg_error = total_error / len(data)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print(f"\n📊 RESULTS:")
    print("-" * 50)
    print(f"Score: {score:.2f}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Median error: ${np.median(errors):.2f}")
    print(f"Max error: ${max(errors):.2f}")
    print(f"Min error: ${min(errors):.2f}")
    
    print(f"\n🎯 ACCURACY BREAKDOWN:")
    print("-" * 50)
    print(f"Perfect (<$0.01): {error_buckets['perfect']} ({error_buckets['perfect']/10:.1f}%)")
    print(f"Excellent (<$0.10): {error_buckets['excellent']} ({error_buckets['excellent']/10:.1f}%)")
    print(f"Great (<$0.50): {error_buckets['great']} ({error_buckets['great']/10:.1f}%)")
    print(f"Good (<$1.00): {error_buckets['good']} ({error_buckets['good']/10:.1f}%)")
    print(f"OK (<$5.00): {error_buckets['ok']} ({error_buckets['ok']/10:.1f}%)")
    print(f"Poor (≥$5.00): {error_buckets['poor']} ({error_buckets['poor']/10:.1f}%)")
    
    print(f"\n⚡ PERFORMANCE:")
    print("-" * 50)
    print(f"Cases improved vs Ultra ML: {improvements}/{len(data)} ({improvements/len(data)*100:.1f}%)")
    print(f"Processing time: {process_time:.1f} seconds ({process_time/len(data)*1000:.1f} ms/case)")
    
    return score

def analyze_remaining_errors():
    """Analyze any remaining significant errors"""
    
    print("\n🔍 ANALYZING REMAINING ERRORS")
    print("=" * 70)
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calculator = UltimateMLCalculator()
    
    # Find cases with error > $1
    significant_errors = []
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = abs(predicted - expected)
        
        if error > 1.0:
            significant_errors.append({
                'index': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': predicted,
                'error': error,
                'receipt_cents': int(round((receipts % 1) * 100))
            })
    
    if significant_errors:
        print(f"Found {len(significant_errors)} cases with error > $1.00")
        
        # Sort by error
        significant_errors.sort(key=lambda x: x['error'], reverse=True)
        
        print("\nTop 10 remaining errors:")
        print("-" * 80)
        print("Idx  Days Miles Receipts  Cents  Expected  Predicted  Error")
        print("-" * 80)
        
        for e in significant_errors[:10]:
            print(f"{e['index']:3d}  {e['days']:4d} {e['miles']:5.0f} {e['receipts']:8.2f}  "
                  f".{e['receipt_cents']:02d}  {e['expected']:8.2f} {e['predicted']:9.2f} {e['error']:6.2f}")
    else:
        print("🎉 No cases with error > $1.00!")

if __name__ == "__main__":
    score = test_score()
    
    previous_best = 852.16
    if score < previous_best:
        improvement = previous_best - score
        print(f"\n🎉 NEW RECORD! Beat {previous_best:.2f} by {improvement:.2f} points!")
        print(f"   That's a {improvement/previous_best*100:.1f}% improvement!")
        
        if score < 100:
            print("\n🏆 ACHIEVED SUB-100 SCORE! We've reached 99%+ accuracy!")
    else:
        print(f"\n😔 Score {score:.2f} doesn't beat our previous best of {previous_best:.2f}")
    
    # Analyze remaining errors
    analyze_remaining_errors() 