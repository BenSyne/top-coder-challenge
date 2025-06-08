#!/usr/bin/env python3
"""Test the ultra ML calculator score"""

import json
from ultra_ml_calculator import UltraMLCalculator
import time

def test_score():
    """Test score on public cases"""
    
    print("🚀 Testing Ultra ML Calculator...")
    print("=" * 60)
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Initialize calculator (will train/load models)
    start_time = time.time()
    calculator = UltraMLCalculator()
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.1f} seconds")
    
    total_error = 0
    exact_matches = 0
    improvements = 0  # Cases where we beat the old model
    
    print("\nProcessing test cases...")
    start_time = time.time()
    
    for i, case in enumerate(data):
        if i % 100 == 0:
            print(f"Progress: {i}/1000")
        
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = abs(predicted - expected)
        total_error += error
        
        if error <= 0.01:
            exact_matches += 1
        
        # Compare with our previous best (roughly)
        if error < 25.82:  # Our previous average
            improvements += 1
    
    process_time = time.time() - start_time
    
    avg_error = total_error / len(data)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print(f"\n📊 RESULTS:")
    print("-" * 40)
    print(f"Score: {score:.2f}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Exact matches: {exact_matches}")
    print(f"Cases improved: {improvements}/{len(data)} ({improvements/len(data)*100:.1f}%)")
    print(f"Processing time: {process_time:.1f} seconds ({process_time/len(data)*1000:.1f} ms/case)")
    
    return score

def compare_with_fast_ml():
    """Compare Ultra ML with Fast ML on specific cases"""
    
    print("\n🔍 COMPARISON WITH FAST ML")
    print("=" * 60)
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Initialize both calculators
    from ultra_ml_calculator import UltraMLCalculator
    from fast_ml_calculator import FastMLCalculator
    
    ultra = UltraMLCalculator()
    fast = FastMLCalculator()
    
    # Test on worst cases from our error analysis
    worst_indices = [413, 229, 477, 550, 732, 564, 760, 180, 918, 746]
    
    print("\nWorst case comparisons:")
    print("-" * 70)
    print("Idx  Days Miles Receipts  Expected   Fast ML  Ultra ML  Improvement")
    print("-" * 70)
    
    total_fast_error = 0
    total_ultra_error = 0
    
    for idx in worst_indices:
        case = data[idx]
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        fast_pred = fast.calculate(days, miles, receipts)
        ultra_pred = ultra.calculate(days, miles, receipts)
        
        fast_error = abs(fast_pred - expected)
        ultra_error = abs(ultra_pred - expected)
        improvement = fast_error - ultra_error
        
        total_fast_error += fast_error
        total_ultra_error += ultra_error
        
        print(f"{idx:3d}  {days:4d} {miles:5.0f} {receipts:7.2f} {expected:8.2f} "
              f"{fast_pred:8.2f} {ultra_pred:8.2f} {improvement:+8.2f}")
    
    print("-" * 70)
    print(f"Average errors - Fast ML: ${total_fast_error/len(worst_indices):.2f}, "
          f"Ultra ML: ${total_ultra_error/len(worst_indices):.2f}")
    print(f"Average improvement: ${(total_fast_error - total_ultra_error)/len(worst_indices):.2f}")

if __name__ == "__main__":
    score = test_score()
    
    previous_best = 2682.29
    if score < previous_best:
        improvement = previous_best - score
        print(f"\n🎉 NEW RECORD! Beat {previous_best:.2f} by {improvement:.2f} points!")
        print(f"   That's a {improvement/previous_best*100:.1f}% improvement!")
    else:
        print(f"\n😔 Score {score:.2f} doesn't beat our previous best of {previous_best:.2f}")
    
    # Run comparison
    compare_with_fast_ml() 