#!/usr/bin/env python3
"""
Strategy for achieving 1000/1000 exact matches
"""

import json
import math
from collections import defaultdict

def analyze_remaining_patterns():
    """Deep analysis of the 998 cases we haven't cracked yet"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Our current best formula
    def current_formula(days, miles, receipts):
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
            # Adjustments
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
        
        # Nickel rounding
        return round(amount * 20) / 20
    
    # Analyze error patterns
    error_by_days = defaultdict(list)
    error_by_miles_range = defaultdict(list)
    error_by_receipts_range = defaultdict(list)
    error_by_combo = defaultdict(list)
    
    close_misses = []  # Cases with very small errors
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = current_formula(days, miles, receipts)
        error = abs(predicted - expected)
        
        if error < 0.50:  # Very close cases
            close_misses.append({
                'case': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': predicted,
                'error': error,
                'ratio': expected / predicted if predicted > 0 else 0
            })
        
        # Categorize errors
        error_by_days[days].append(error)
        
        mile_range = f"{(miles//100)*100}-{(miles//100)*100+99}"
        error_by_miles_range[mile_range].append(error)
        
        receipt_range = f"{(receipts//500)*500}-{(receipts//500)*500+499}"
        error_by_receipts_range[receipt_range].append(error)
        
        combo_key = f"{days}d_{mile_range}mi_{receipt_range}r"
        error_by_combo[combo_key].append(error)
    
    print("CLOSE MISS ANALYSIS (Error < $0.50)")
    print("=" * 60)
    print(f"Found {len(close_misses)} cases within $0.50")
    
    # Sort by error
    close_misses.sort(key=lambda x: x['error'])
    
    print("\nClosest misses:")
    for miss in close_misses[:20]:
        print(f"Case {miss['case']:3d}: {miss['days']:2d}d, {miss['miles']:4.0f}mi, ${miss['receipts']:7.2f}")
        print(f"          Expected: ${miss['expected']:7.2f}, Got: ${miss['predicted']:7.2f}, Error: ${miss['error']:.3f}, Ratio: {miss['ratio']:.6f}")
    
    # Look for ratio patterns
    ratios = [miss['ratio'] for miss in close_misses if miss['ratio'] > 0]
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\nAverage ratio (expected/predicted): {avg_ratio:.6f}")
        
        # Common ratios
        ratio_groups = defaultdict(list)
        for miss in close_misses:
            if miss['ratio'] > 0:
                ratio_key = round(miss['ratio'], 3)
                ratio_groups[ratio_key].append(miss)
        
        print("\nMost common ratios:")
        for ratio, group in sorted(ratio_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            if len(group) > 1:
                print(f"  {ratio:.3f}: {len(group)} cases")

def test_advanced_rounding_methods():
    """Test more sophisticated rounding methods from 1960s"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # 1960s-era rounding methods
    rounding_methods = [
        ("Banker's (round half to even)", lambda x: round(x * 100) / 100),
        ("Always round up", lambda x: math.ceil(x * 100) / 100),
        ("Round to nearest penny", lambda x: round(x, 2)),
        ("Round to nearest nickel", lambda x: round(x * 20) / 20),
        ("Round to nearest dime", lambda x: round(x * 10) / 10),
        ("Round to nearest quarter", lambda x: round(x * 4) / 4),
        ("Truncate to penny", lambda x: math.floor(x * 100) / 100),
        ("Round to nearest $0.05 up", lambda x: math.ceil(x * 20) / 20),
        ("Round to nearest $0.05 down", lambda x: math.floor(x * 20) / 20),
        ("1960s accounting round", lambda x: math.floor(x * 100 + 0.5) / 100),
    ]
    
    print("\n\nADVANCED ROUNDING METHOD TESTING")
    print("=" * 60)
    
    for method_name, round_func in rounding_methods:
        exact_matches = 0
        total_error = 0
        
        for case in data:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            # Use our base calculation
            base = calculate_base_amount(days, miles, receipts)
            predicted = max(0.0, round_func(base))
            
            error = abs(predicted - expected)
            total_error += error
            
            if error <= 0.01:
                exact_matches += 1
        
        avg_error = total_error / len(data)
        print(f"{method_name:25}: {exact_matches:3d} exact, ${avg_error:7.2f} avg error")

def calculate_base_amount(days, miles, receipts):
    """Our best base calculation before rounding"""
    # Apply caps
    capped_receipts = receipts
    if receipts > 1800:
        capped_receipts = 1800 + (receipts - 1800) * 0.15
    
    capped_miles = miles
    if miles > 800:
        capped_miles = 800 + (miles - 800) * 0.25
    
    # Base calculation with current best coefficients
    amount = 266.71 + 50.05 * days + 0.4456 * capped_miles + 0.3829 * capped_receipts
    
    # Check for rounding bug
    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
        amount *= 0.457
    else:
        # Apply other adjustments
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

def suggest_next_steps():
    """Suggest strategies for reaching 1000/1000 exact matches"""
    
    print("\n\nNEXT STEPS FOR COMPLETE REVERSE ENGINEERING")
    print("=" * 60)
    
    strategies = [
        "1. MACHINE LEARNING APPROACH",
        "   - Train gradient boosting model on residual errors",
        "   - Use neural network to capture complex interactions",
        "   - Random forest to find additional rules",
        "",
        "2. HISTORICAL ROUNDING RESEARCH", 
        "   - Research 1960s IBM/accounting system rounding",
        "   - Test vintage floating-point representations",
        "   - Check for fixed-point arithmetic artifacts",
        "",
        "3. SYSTEMATIC RULE MINING",
        "   - Decision tree on close-miss cases",
        "   - Pattern detection on specific day/mile/receipt ranges",
        "   - Test for additional caps or thresholds",
        "",
        "4. BRUTE FORCE COEFFICIENT OPTIMIZATION",
        "   - Grid search with higher precision",
        "   - Genetic algorithm for coefficient evolution", 
        "   - Bayesian optimization for efficient search",
        "",
        "5. LOOKUP TABLE DETECTION",
        "   - Check if some ranges use exact lookup values",
        "   - Test for discrete jumps in reimbursement",
        "   - Historical rate tables from 1960s",
    ]
    
    for strategy in strategies:
        print(strategy)

if __name__ == "__main__":
    analyze_remaining_patterns()
    test_advanced_rounding_methods()
    suggest_next_steps() 