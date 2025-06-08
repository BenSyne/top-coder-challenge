#!/usr/bin/env python3
"""
Deeper analysis to understand the true calculation patterns
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_and_prepare_data():
    """Load and prepare the data for analysis"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame([{
        'days': case['input']['trip_duration_days'],
        'miles': case['input']['miles_traveled'],
        'receipts': case['input']['total_receipts_amount'],
        'expected': case['expected_output']
    } for case in data])
    
    # Add derived features
    df['miles_per_day'] = df['miles'] / df['days']
    df['receipts_per_day'] = df['receipts'] / df['days']
    
    return df

def analyze_base_component(df):
    """Try to isolate the base per diem component"""
    print("Analyzing base per diem component...")
    print("-" * 60)
    
    # Look at trips with minimal miles and receipts
    minimal_trips = df[(df['miles'] < 50) & (df['receipts'] < 50)]
    
    for days in range(1, 8):
        subset = minimal_trips[minimal_trips['days'] == days]
        if len(subset) > 0:
            avg_reimb = subset['expected'].mean()
            print(f"{days} days (minimal miles/receipts): ${avg_reimb:.2f} total, ${avg_reimb/days:.2f}/day")

def analyze_mileage_component(df):
    """Analyze mileage reimbursement patterns"""
    print("\n\nAnalyzing mileage component...")
    print("-" * 60)
    
    # Look at 1-day trips with varying miles and minimal receipts
    one_day_minimal_receipts = df[(df['days'] == 1) & (df['receipts'] < 30)]
    
    if len(one_day_minimal_receipts) > 5:
        # Simple linear regression to find mileage rate
        X = one_day_minimal_receipts['miles'].values.reshape(-1, 1)
        y = one_day_minimal_receipts['expected'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        print(f"Linear regression on 1-day trips with minimal receipts:")
        print(f"  Intercept (base): ${model.intercept_:.2f}")
        print(f"  Miles coefficient: ${model.coef_[0]:.4f}/mile")
    
    # Analyze by mileage ranges
    print("\nAverage reimbursement by mileage range (1-day trips):")
    one_day = df[df['days'] == 1]
    mile_ranges = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 1000)]
    
    for min_m, max_m in mile_ranges:
        subset = one_day[(one_day['miles'] >= min_m) & (one_day['miles'] < max_m)]
        if len(subset) > 0:
            avg_reimb = subset['expected'].mean()
            avg_miles = subset['miles'].mean()
            print(f"  {min_m}-{max_m} miles: ${avg_reimb:.2f} avg total (avg {avg_miles:.0f} miles)")

def analyze_receipt_component(df):
    """Analyze receipt reimbursement patterns"""
    print("\n\nAnalyzing receipt component...")
    print("-" * 60)
    
    # Look at trips with minimal miles to isolate receipt effect
    minimal_miles = df[df['miles'] < 50]
    
    receipt_ranges = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 2500)]
    
    print("Average reimbursement by receipt range (low-mileage trips):")
    for min_r, max_r in receipt_ranges:
        subset = minimal_miles[(minimal_miles['receipts'] >= min_r) & (minimal_miles['receipts'] < max_r)]
        if len(subset) > 0:
            avg_reimb = subset['expected'].mean()
            avg_receipts = subset['receipts'].mean()
            avg_days = subset['days'].mean()
            print(f"  ${min_r}-${max_r}: ${avg_reimb:.2f} total (avg ${avg_receipts:.0f} receipts, {avg_days:.1f} days)")

def find_simple_pattern(df):
    """Try to find a simple pattern that explains most of the variance"""
    print("\n\nSearching for simple pattern...")
    print("-" * 60)
    
    # Multiple linear regression
    from sklearn.preprocessing import PolynomialFeatures
    
    # Try simple linear model first
    X = df[['days', 'miles', 'receipts']].values
    y = df['expected'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("Simple linear regression:")
    print(f"  Intercept: ${model.intercept_:.2f}")
    print(f"  Days coefficient: ${model.coef_[0]:.2f}")
    print(f"  Miles coefficient: ${model.coef_[1]:.4f}")
    print(f"  Receipts coefficient: ${model.coef_[2]:.4f}")
    
    # Calculate R-squared
    predictions = model.predict(X)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R-squared: {r_squared:.4f}")
    
    # Test on our problem cases
    test_cases = [
        (3, 93, 1.42),
        (1, 55, 3.6),
        (5, 130, 306.90)
    ]
    
    print("\nPredictions on test cases:")
    for days, miles, receipts in test_cases:
        pred = model.predict([[days, miles, receipts]])[0]
        actual_idx = df[(df['days'] == days) & (df['miles'] == miles) & 
                       (abs(df['receipts'] - receipts) < 0.01)].index
        if len(actual_idx) > 0:
            actual = df.loc[actual_idx[0], 'expected']
            print(f"  {days}d, {miles}mi, ${receipts:.2f}: Predicted ${pred:.2f}, Actual ${actual:.2f}")

if __name__ == "__main__":
    df = load_and_prepare_data()
    analyze_base_component(df)
    analyze_mileage_component(df)
    analyze_receipt_component(df)
    find_simple_pattern(df) 