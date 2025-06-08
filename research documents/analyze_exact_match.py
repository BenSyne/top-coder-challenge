#!/usr/bin/env python3
"""
Deep dive into the exact match case
"""

import json
import math

def analyze_exact_match():
    """Analyze Case 198 that gives us an exact match"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Case 198: 1d, 420mi, $2273.60 → $1220.35
    case = data[198]
    days = case['input']['trip_duration_days']
    miles = case['input']['miles_traveled'] 
    receipts = case['input']['total_receipts_amount']
    expected = case['expected_output']
    
    print("EXACT MATCH ANALYSIS - Case 198")
    print("=" * 50)
    print(f"Input: {days} days, {miles} miles, ${receipts:.2f} receipts")
    print(f"Expected output: ${expected:.2f}")
    
    # Step by step calculation
    print("\nStep-by-step calculation:")
    
    # 1. Check for caps
    capped_receipts = receipts
    if receipts > 1800:
        capped_receipts = 1800 + (receipts - 1800) * 0.15
        print(f"1. Receipt cap: ${receipts:.2f} → ${capped_receipts:.2f}")
    else:
        print(f"1. No receipt cap needed (${receipts:.2f} ≤ $1800)")
    
    capped_miles = miles
    if miles > 800:
        capped_miles = 800 + (miles - 800) * 0.25
        print(f"2. Mile cap: {miles} → {capped_miles:.2f}")
    else:
        print(f"2. No mile cap needed ({miles} ≤ 800)")
    
    # 3. Base calculation
    base = 266.71 + 50.05 * days + 0.4456 * capped_miles + 0.3829 * capped_receipts
    print(f"3. Base: 266.71 + 50.05×{days} + 0.4456×{capped_miles} + 0.3829×{capped_receipts:.2f} = ${base:.4f}")
    
    # 4. Check rounding bug
    receipt_str = f"{receipts:.2f}"
    has_rounding_bug = receipt_str.endswith('.49') or receipt_str.endswith('.99')
    if has_rounding_bug:
        base *= 0.457
        print(f"4. Rounding bug applied: ${base:.4f}")
    else:
        print("4. No rounding bug")
    
    # 5. Other adjustments
    adjustments = []
    
    if days == 5:
        base *= 0.92
        adjustments.append("5-day penalty")
    
    if days >= 12 and miles >= 1000:
        base *= 0.85
        adjustments.append("long trip + high mileage penalty")
    
    if days <= 2 and miles < 100 and receipts < 50:
        base *= 1.15
        adjustments.append("short minimal trip boost")
    
    if 7 <= days <= 8 and miles >= 1000 and receipts >= 1000:
        base *= 1.25
        adjustments.append("7-8 day high value boost")
    
    if 13 <= days <= 14 and miles >= 1000 and receipts >= 1000:
        base *= 1.20
        adjustments.append("13-14 day high value boost")
    
    miles_per_day = miles / days if days > 0 else miles
    if 180 <= miles_per_day <= 220:
        base += 30.0
        adjustments.append("efficiency bonus")
    
    if adjustments:
        print(f"5. Adjustments applied: {', '.join(adjustments)}")
        print(f"   Result: ${base:.4f}")
    else:
        print("5. No adjustments applied")
    
    # 6. Final rounding
    ceiling_result = math.ceil(base * 100) / 100
    nickel_result = round(base * 20) / 20
    standard_result = round(base, 2)
    
    print(f"\n6. Final rounding options:")
    print(f"   Standard: ${standard_result:.2f} (error: ${abs(standard_result - expected):.4f})")
    print(f"   Ceiling:  ${ceiling_result:.2f} (error: ${abs(ceiling_result - expected):.4f})")
    print(f"   Nickel:   ${nickel_result:.2f} (error: ${abs(nickel_result - expected):.4f})")
    
    if abs(ceiling_result - expected) <= 0.01:
        print(f"\n🎯 EXACT MATCH with ceiling rounding!")
    
    # What makes this case special?
    print(f"\nWhat makes this case special:")
    print(f"- Miles per day: {miles_per_day:.1f}")
    print(f"- Receipts per day: ${receipts/days:.2f}")
    print(f"- Receipt amount ends in: .{receipt_str.split('.')[-1]}")
    print(f"- No major adjustments applied")

def look_for_similar_cases():
    """Look for other cases similar to our exact match"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("\n\nLOOKING FOR SIMILAR CASES")
    print("=" * 50)
    
    # Characteristics of Case 198: 1 day, ~400 miles, high receipts
    similar_cases = []
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        # Look for 1-day trips with 300-500 miles and high receipts
        if (days == 1 and 
            300 <= miles <= 500 and 
            receipts > 2000):
            
            similar_cases.append({
                'case': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected
            })
    
    print(f"Found {len(similar_cases)} similar cases (1 day, 300-500 miles, >$2000 receipts):")
    
    for case in similar_cases[:10]:
        print(f"  Case {case['case']}: {case['days']}d, {case['miles']}mi, ${case['receipts']:.2f} → ${case['expected']:.2f}")

if __name__ == "__main__":
    analyze_exact_match()
    look_for_similar_cases() 