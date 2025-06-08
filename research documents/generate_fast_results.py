#!/usr/bin/env python3
"""
Fast batch processor for generating private results
Uses the FastMLCalculator directly instead of calling shell commands
"""

import json
import sys
from fast_ml_calculator import FastMLCalculator

def generate_results():
    """Generate results for all private test cases"""
    
    print("🚀 Fast Batch Results Generator")
    print("=" * 50)
    
    # Load private test cases
    print("Loading private test cases...")
    with open('private_cases.json', 'r') as f:
        private_cases = json.load(f)
    
    print(f"Found {len(private_cases)} test cases")
    
    # Initialize calculator (will load from cache)
    print("Loading ML models from cache...")
    calculator = FastMLCalculator()
    
    # Process all cases
    print("Processing test cases...")
    results = []
    
    for i, case in enumerate(private_cases):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(private_cases)} ({i/len(private_cases)*100:.1f}%)")
        
        days = case['trip_duration_days']
        miles = case['miles_traveled']
        receipts = case['total_receipts_amount']
        
        result = calculator.calculate(days, miles, receipts)
        results.append(f"{result:.2f}")
    
    # Write results
    print("\nWriting results to private_results.txt...")
    with open('private_results.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    print(f"✅ Successfully generated {len(results)} results!")
    print("📄 Results saved to private_results.txt")
    
    return len(results)

if __name__ == "__main__":
    try:
        count = generate_results()
        print(f"\n🎉 Complete! Generated {count} results.")
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1) 