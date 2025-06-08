#!/usr/bin/env python3

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultimate_ml_calculator import UltimateMLCalculator

def main():
    # Load the calculator once
    print("Loading Ultimate ML Calculator...", file=sys.stderr)
    calculator = UltimateMLCalculator()
    
    # Load test cases
    print("Loading test cases...", file=sys.stderr)
    with open('private_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    print(f"Processing {len(test_cases)} cases...", file=sys.stderr)
    
    # Process all cases
    results = []
    for i, case in enumerate(test_cases):
        if i % 500 == 0 and i > 0:
            print(f"Progress: {i}/{len(test_cases)} cases processed...", file=sys.stderr)
        
        try:
            result = calculator.calculate(
                case['trip_duration_days'],
                case['miles_traveled'], 
                case['total_receipts_amount']
            )
            results.append(f"{result:.2f}")
        except Exception as e:
            print(f"Error on case {i+1}: {e}", file=sys.stderr)
            results.append("ERROR")
    
    # Output results
    print("Writing results...", file=sys.stderr)
    with open('private_results.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    print(f"✅ Completed! Processed {len(results)} cases.", file=sys.stderr)

if __name__ == "__main__":
    main() 