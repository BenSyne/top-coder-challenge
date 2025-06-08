#!/usr/bin/env python3
"""
Analyze if the system uses discrete rules or lookup tables
"""

import json
import pandas as pd
import numpy as np

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame([{
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'],
        'receipts': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    } for case in data])
    
    return df

def analyze_base_rates(df):
    """Check if there are fixed base rates per day"""
    print("ANALYZING BASE RATES")
    print("=" * 60)
    
    # Look at trips with very minimal miles and receipts
    minimal = df[(df['miles'] < 20) & (df['receipts'] < 10)]
    
    print("Trips with minimal miles (<20) and receipts (<$10):")
    print(f"{'Days':>4} {'Miles':>6} {'Receipts':>8} {'Expected':>10} {'Per Day':>10}")
    print("-" * 50)
    
    for _, row in minimal.iterrows():
        per_day = row['expected'] / row['days']
        print(f"{row['days']:4.0f} {row['miles']:6.0f} {row['receipts']:8.2f} "
              f"{row['expected']:10.2f} {per_day:10.2f}")

def analyze_mileage_rates(df):
    """Check if mileage follows fixed rates or tiers"""
    print("\n\nANALYZING MILEAGE RATES")
    print("=" * 60)
    
    # Look at 1-day trips with minimal receipts to isolate mileage effect
    one_day_minimal = df[(df['days'] == 1) & (df['receipts'] < 10)]
    one_day_minimal = one_day_minimal.sort_values('miles')
    
    print("1-day trips with minimal receipts (<$10):")
    print(f"{'Miles':>6} {'Receipts':>8} {'Expected':>10} {'Implied $/mile':>15}")
    print("-" * 50)
    
    for _, row in one_day_minimal.head(20).iterrows():
        # Subtract assumed base (100) to get mileage portion
        mileage_portion = row['expected'] - 100
        rate_per_mile = mileage_portion / row['miles'] if row['miles'] > 0 else 0
        print(f"{row['miles']:6.0f} {row['receipts']:8.2f} {row['expected']:10.2f} "
              f"{rate_per_mile:15.4f}")

def analyze_receipt_patterns(df):
    """Look for discrete receipt processing rules"""
    print("\n\nANALYZING RECEIPT PATTERNS")
    print("=" * 60)
    
    # Group by receipt ranges
    receipt_ranges = [
        (0, 10), (10, 50), (50, 100), (100, 200), (200, 500),
        (500, 1000), (1000, 1500), (1500, 2500)
    ]
    
    print("Average reimbursement rate by receipt range:")
    print(f"{'Receipt Range':>20} {'Avg Rate':>10} {'Count':>8}")
    print("-" * 40)
    
    for min_r, max_r in receipt_ranges:
        subset = df[(df['receipts'] >= min_r) & (df['receipts'] < max_r)]
        if len(subset) > 0:
            # Calculate implied receipt reimbursement rate
            # Subtract base per diem and estimated mileage
            subset['base_estimate'] = subset['days'] * 100 + subset['miles'] * 0.45
            subset['receipt_portion'] = subset['expected'] - subset['base_estimate']
            subset['receipt_rate'] = subset['receipt_portion'] / subset['receipts']
            
            avg_rate = subset['receipt_rate'].mean()
            print(f"${min_r:4d} - ${max_r:4d} {avg_rate:10.3f} {len(subset):8d}")

def find_exact_matches_pattern(df):
    """Look for patterns that might lead to exact matches"""
    print("\n\nSEARCHING FOR EXACT MATCH PATTERNS")
    print("=" * 60)
    
    # Try simple formulas with rounding
    formulas = [
        ("50*d + 0.55*m + 0.35*r", lambda d, m, r: 50*d + 0.55*m + 0.35*r),
        ("100*d + 0.45*m + 0.45*r", lambda d, m, r: 100*d + 0.45*m + 0.45*r),
        ("75*d + 0.50*m + 0.40*r", lambda d, m, r: 75*d + 0.50*m + 0.40*r),
        ("ceil(100*d + 0.5*(m+r))", lambda d, m, r: np.ceil(100*d + 0.5*(m+r))),
        ("round(100*d + 0.45*m + 0.45*r, 2)", lambda d, m, r: round(100*d + 0.45*m + 0.45*r, 2)),
    ]
    
    for name, formula in formulas:
        exact_matches = 0
        close_matches = 0
        total_error = 0
        
        for _, row in df.iterrows():
            predicted = formula(row['days'], row['miles'], row['receipts'])
            error = abs(predicted - row['expected'])
            total_error += error
            
            if error < 0.01:
                exact_matches += 1
            if error < 1.00:
                close_matches += 1
        
        avg_error = total_error / len(df)
        print(f"\n{name}:")
        print(f"  Exact matches: {exact_matches}")
        print(f"  Close matches: {close_matches}")
        print(f"  Average error: ${avg_error:.2f}")

def analyze_trip_categories(df):
    """See if trips fall into discrete categories"""
    print("\n\nANALYZING TRIP CATEGORIES")
    print("=" * 60)
    
    # Define potential categories
    df['miles_per_day'] = df['miles'] / df['days']
    df['receipts_per_day'] = df['receipts'] / df['days']
    
    # Category 1: Short local trips
    cat1 = df[(df['days'] <= 2) & (df['miles'] < 100)]
    print(f"Short local trips (≤2 days, <100 miles): {len(cat1)} cases")
    print(f"  Average reimbursement: ${cat1['expected'].mean():.2f}")
    
    # Category 2: Standard business trips
    cat2 = df[(df['days'].between(3, 7)) & (df['miles'].between(100, 500))]
    print(f"\nStandard trips (3-7 days, 100-500 miles): {len(cat2)} cases")
    print(f"  Average reimbursement: ${cat2['expected'].mean():.2f}")
    
    # Category 3: Long trips
    cat3 = df[df['days'] >= 8]
    print(f"\nLong trips (≥8 days): {len(cat3)} cases")
    print(f"  Average reimbursement: ${cat3['expected'].mean():.2f}")
    
    # Category 4: High mileage trips
    cat4 = df[df['miles'] >= 800]
    print(f"\nHigh mileage trips (≥800 miles): {len(cat4)} cases")
    print(f"  Average reimbursement: ${cat4['expected'].mean():.2f}")

def check_for_lookup_table(df):
    """Check if certain combinations always give the same result"""
    print("\n\nCHECKING FOR LOOKUP TABLE PATTERNS")
    print("=" * 60)
    
    # Group by exact input combinations
    grouped = df.groupby(['days', 'miles', 'receipts'])['expected'].agg(['count', 'mean', 'std'])
    
    # Find combinations that appear multiple times
    duplicates = grouped[grouped['count'] > 1]
    
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} input combinations that appear multiple times:")
        print(duplicates)
    else:
        print("No duplicate input combinations found - not a simple lookup table")
    
    # Check if similar inputs give similar outputs
    print("\nChecking consistency for similar inputs:")
    
    # Find cases with very similar inputs
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        similar = df[
            (abs(df['days'] - row['days']) <= 0) &
            (abs(df['miles'] - row['miles']) <= 5) &
            (abs(df['receipts'] - row['receipts']) <= 5) &
            (df.index != i)
        ]
        
        if len(similar) > 0:
            print(f"\nBase case: {row['days']}d, {row['miles']}mi, ${row['receipts']:.2f} → ${row['expected']:.2f}")
            for _, sim in similar.iterrows():
                diff = sim['expected'] - row['expected']
                print(f"  Similar: {sim['days']}d, {sim['miles']}mi, ${sim['receipts']:.2f} → ${sim['expected']:.2f} (diff: ${diff:.2f})")

if __name__ == "__main__":
    df = load_data()
    
    analyze_base_rates(df)
    analyze_mileage_rates(df)
    analyze_receipt_patterns(df)
    find_exact_matches_pattern(df)
    analyze_trip_categories(df)
    check_for_lookup_table(df) 