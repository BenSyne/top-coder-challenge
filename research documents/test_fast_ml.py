#!/usr/bin/env python3
"""Test the fast ML calculator score"""

import json
from fast_ml_calculator import FastMLCalculator

def test_score():
    """Test score on public cases"""
    
    print("Testing Fast ML Calculator...")
    
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Initialize calculator (will load from cache)
    calculator = FastMLCalculator()
    
    total_error = 0
    exact_matches = 0
    
    print("Processing test cases...")
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
    
    avg_error = total_error / len(data)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print(f"\nResults:")
    print(f"Score: {score:.2f}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Exact matches: {exact_matches}")
    
    return score

if __name__ == "__main__":
    score = test_score()
    
    if score < 8800:
        print(f"\n🎉 SUCCESS! Beat target of 8,800 with score {score:.2f}")
    else:
        print(f"\n😔 Score {score:.2f} doesn't beat target of 8,800") 