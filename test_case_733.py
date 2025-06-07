#!/usr/bin/env python3
"""
Test Case 733 which is similar to our exact match
"""

import math

def test_case_733():
    """Test Case 733: 1d, 462mi, $2047.57 → $1202.90"""
    
    days, miles, receipts = 1, 462, 2047.57
    expected = 1202.90
    
    print("TESTING CASE 733")
    print("=" * 40)
    print(f"Input: {days} days, {miles} miles, ${receipts:.2f} receipts")
    print(f"Expected: ${expected:.2f}")
    
    # Apply receipt cap
    capped_receipts = receipts
    if receipts > 1800:
        capped_receipts = 1800 + (receipts - 1800) * 0.15
        print(f"Receipt cap: ${receipts:.2f} → ${capped_receipts:.2f}")
    
    # No mile cap needed (462 ≤ 800)
    capped_miles = miles
    
    # Base calculation
    base = 266.71 + 50.05 * days + 0.4456 * capped_miles + 0.3829 * capped_receipts
    print(f"Base: ${base:.6f}")
    
    # Check rounding bug
    receipt_str = f"{receipts:.2f}"
    if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
        print("Rounding bug detected!")
        base *= 0.457
        print(f"After rounding bug: ${base:.6f}")
    else:
        print("No rounding bug")
    
    # Test different rounding
    standard = round(base, 2)
    ceiling = math.ceil(base * 100) / 100
    nickel = round(base * 20) / 20
    
    print(f"\nRounding options:")
    print(f"Standard: ${standard:.2f} (error: ${abs(standard - expected):.4f})")
    print(f"Ceiling:  ${ceiling:.2f} (error: ${abs(ceiling - expected):.4f})")
    print(f"Nickel:   ${nickel:.2f} (error: ${abs(nickel - expected):.4f})")
    
    if abs(nickel - expected) <= 0.01:
        print(f"🎯 EXACT MATCH with nickel rounding!")
    elif abs(ceiling - expected) <= 0.01:
        print(f"🎯 EXACT MATCH with ceiling rounding!")
    elif abs(standard - expected) <= 0.01:
        print(f"🎯 EXACT MATCH with standard rounding!")

def check_rounding_bug_cases():
    """Check cases with rounding bug to see if we can get exact matches"""
    
    import json
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("\n\nCHECKING ROUNDING BUG CASES")
    print("=" * 40)
    
    rounding_bug_cases = []
    
    for i, case in enumerate(data):
        receipts = case['input']['total_receipts_amount']
        receipt_str = f"{receipts:.2f}"
        
        if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
            rounding_bug_cases.append((i, case))
    
    print(f"Found {len(rounding_bug_cases)} rounding bug cases")
    
    # Test first few with our formula
    for i, (case_idx, case) in enumerate(rounding_bug_cases[:5]):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Apply our formula
        capped_receipts = receipts
        if receipts > 1800:
            capped_receipts = 1800 + (receipts - 1800) * 0.15
        
        capped_miles = miles
        if miles > 800:
            capped_miles = 800 + (miles - 800) * 0.25
        
        base = 266.71 + 50.05 * days + 0.4456 * capped_miles + 0.3829 * capped_receipts
        base *= 0.457  # Rounding bug
        
        nickel = round(base * 20) / 20
        error = abs(nickel - expected)
        
        print(f"Case {case_idx}: {days}d, {miles}mi, ${receipts:.2f} → Expected: ${expected:.2f}, Got: ${nickel:.2f}, Error: ${error:.4f}")
        
        if error <= 0.01:
            print(f"  🎯 EXACT MATCH!")

if __name__ == "__main__":
    test_case_733()
    check_rounding_bug_cases() 