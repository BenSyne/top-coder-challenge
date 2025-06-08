#!/usr/bin/env python3
"""
Ultra fast batch processor for generating private results
Uses the UltraMLCalculator directly
"""

import json
import sys
from ultra_ml_calculator import UltraMLCalculator
import time

def generate_results():
    """Generate results for all private test cases"""
    
    print("🚀 Ultra ML Batch Results Generator")
    print("=" * 50)
    
    # Load private test cases
    print("Loading private test cases...")
    with open('private_cases.json', 'r') as f:
        private_cases = json.load(f)
    
    print(f"Found {len(private_cases)} test cases")
    
    # Initialize calculator (will load from cache)
    print("Loading Ultra ML models from cache...")
    start_time = time.time()
    calculator = UltraMLCalculator()
    load_time = time.time() - start_time
    print(f"Models loaded in {load_time:.1f} seconds")
    
    # Process all cases
    print("\nProcessing test cases...")
    results = []
    start_time = time.time()
    
    for i, case in enumerate(private_cases):
        if i % 250 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                rate = i / elapsed
                eta = (len(private_cases) - i) / rate
                print(f"Progress: {i}/{len(private_cases)} ({i/len(private_cases)*100:.1f}%) "
                      f"- {rate:.0f} cases/sec - ETA: {eta:.0f}s")
            else:
                print(f"Progress: {i}/{len(private_cases)} (0.0%)")
        
        days = case['trip_duration_days']
        miles = case['miles_traveled']
        receipts = case['total_receipts_amount']
        
        result = calculator.calculate(days, miles, receipts)
        results.append(f"{result:.2f}")
    
    process_time = time.time() - start_time
    
    # Write results
    print(f"\nWriting results to private_results.txt...")
    with open('private_results.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    print(f"\n✅ Successfully generated {len(results)} results!")
    print(f"📄 Results saved to private_results.txt")
    print(f"⏱️  Total processing time: {process_time:.1f} seconds")
    print(f"⚡ Average: {process_time/len(results)*1000:.1f} ms/case")
    
    return len(results)

if __name__ == "__main__":
    try:
        count = generate_results()
        print(f"\n🏆 Ultra ML processing complete! Generated {count} results.")
        print(f"📈 Expected score: ~852 (based on public test performance)")
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1) 