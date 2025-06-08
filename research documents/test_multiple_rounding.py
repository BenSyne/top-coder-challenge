#!/usr/bin/env python3
"""
Test multiple rounding strategies
"""

import json
import math

def base_calculate(days, miles, receipts):
    """Our best base calculation"""
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

def test_rounding_strategies():
    """Test various rounding strategies"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    strategies = [
        ("Standard", lambda x: round(x, 2)),
        ("Nickel ($0.05)", lambda x: round(x * 20) / 20),
        ("Dime ($0.10)", lambda x: round(x * 10) / 10),
        ("Quarter ($0.25)", lambda x: round(x * 4) / 4),
        ("Half dollar ($0.50)", lambda x: round(x * 2) / 2),
        ("Dollar", lambda x: round(x)),
        ("Ceiling", lambda x: math.ceil(x * 100) / 100),
        ("Floor", lambda x: math.floor(x * 100) / 100),
        ("Banker's rounding", lambda x: round(x * 100) / 100),
        ("Round to nearest $5", lambda x: round(x / 5) * 5),
        ("Round to nearest $10", lambda x: round(x / 10) * 10),
    ]
    
    results = []
    
    for strategy_name, round_func in strategies:
        exact_matches = 0
        close_matches = 0
        total_error = 0
        
        exact_cases = []
        
        for i, case in enumerate(data):
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            base_amount = base_calculate(days, miles, receipts)
            predicted = max(0.0, round_func(base_amount))
            
            error = abs(predicted - expected)
            total_error += error
            
            if error <= 0.01:
                exact_matches += 1
                exact_cases.append(i)
            
            if error <= 1.00:
                close_matches += 1
        
        avg_error = total_error / len(data)
        
        results.append({
            'strategy': strategy_name,
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'avg_error': avg_error,
            'score': total_error,
            'exact_cases': exact_cases
        })
    
    # Sort by exact matches, then by average error
    results.sort(key=lambda x: (-x['exact_matches'], x['avg_error']))
    
    print("ROUNDING STRATEGY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Exact':>6} {'Close':>6} {'Avg Error':>10} {'Score':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['strategy']:<20} {r['exact_matches']:>6} {r['close_matches']:>6} "
              f"${r['avg_error']:>9.2f} {r['score']:>10.0f}")
    
    # Show details of best strategy
    best = results[0]
    if best['exact_matches'] > 0:
        print(f"\n🎯 BEST STRATEGY: {best['strategy']}")
        print(f"Found {best['exact_matches']} exact matches in cases: {best['exact_cases']}")
        
        # Show the exact match cases
        print("\nExact match details:")
        for case_idx in best['exact_cases'][:5]:  # Show first 5
            case = data[case_idx]
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            print(f"  Case {case_idx}: {days}d, {miles}mi, ${receipts:.2f} → ${expected:.2f}")

if __name__ == "__main__":
    test_rounding_strategies() 