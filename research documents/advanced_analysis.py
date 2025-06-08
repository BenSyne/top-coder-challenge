#!/usr/bin/env python3
"""
Advanced analysis to understand non-linear patterns
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

def load_data():
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
    df['total_inputs'] = df['days'] + df['miles'] + df['receipts']
    
    return df

def analyze_long_trips(df):
    """Analyze patterns in longer trips"""
    print("Analyzing long trips (7+ days)...")
    print("-" * 60)
    
    long_trips = df[df['days'] >= 7]
    print(f"Number of long trips: {len(long_trips)}")
    print(f"Average reimbursement: ${long_trips['expected'].mean():.2f}")
    print(f"Average miles: {long_trips['miles'].mean():.0f}")
    print(f"Average receipts: ${long_trips['receipts'].mean():.2f}")
    
    # Analyze by mileage level
    print("\nLong trips by mileage level:")
    for min_m in [0, 500, 800, 1000]:
        max_m = min_m + 300 if min_m < 1000 else 2000
        subset = long_trips[(long_trips['miles'] >= min_m) & (long_trips['miles'] < max_m)]
        if len(subset) > 0:
            print(f"  {min_m}-{max_m} miles: ${subset['expected'].mean():.2f} avg (n={len(subset)})")

def analyze_interaction_effects(df):
    """Analyze interaction effects between variables"""
    print("\n\nAnalyzing interaction effects...")
    print("-" * 60)
    
    # Create interaction features
    df['days_miles'] = df['days'] * df['miles']
    df['days_receipts'] = df['days'] * df['receipts']
    df['miles_receipts'] = df['miles'] * df['receipts']
    
    # Fit model with interactions
    features = ['days', 'miles', 'receipts', 'days_miles', 'days_receipts', 'miles_receipts']
    X = df[features].values
    y = df['expected'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("Linear model with interactions:")
    for i, feat in enumerate(features):
        print(f"  {feat}: {model.coef_[i]:.6f}")
    print(f"  Intercept: {model.intercept_:.2f}")
    
    # Calculate R-squared
    predictions = model.predict(X)
    r_squared = model.score(X, y)
    print(f"  R-squared: {r_squared:.4f}")
    
    return model, features

def analyze_polynomial_effects(df):
    """Try polynomial features"""
    print("\n\nAnalyzing polynomial effects...")
    print("-" * 60)
    
    # Simple quadratic model
    X = df[['days', 'miles', 'receipts']].values
    y = df['expected'].values
    
    # Create polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    print(f"Polynomial model (degree 2):")
    print(f"  Number of features: {X_poly.shape[1]}")
    print(f"  R-squared: {model.score(X_poly, y):.4f}")
    
    # Get feature names and show most important
    feature_names = poly.get_feature_names_out(['days', 'miles', 'receipts'])
    coef_importance = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nMost important features:")
    for name, coef in coef_importance[:10]:
        print(f"  {name}: {coef:.6f}")
    
    return model, poly

def test_models_on_problem_cases(df, models):
    """Test models on the problem cases"""
    print("\n\nTesting on problem cases...")
    print("-" * 60)
    
    # Problem cases from evaluation
    problem_cases = [
        (7, 1006, 1181.33, 2279.82),
        (8, 1025, 1031.33, 2214.64),
        (7, 1033, 1013.03, 2119.83),
        (14, 1020, 1201.75, 2337.73),
        # Also test on simple cases
        (3, 93, 1.42, 364.51),
        (1, 55, 3.6, 126.06)
    ]
    
    linear_model, interaction_model, poly_model, poly_transformer = models
    
    for days, miles, receipts, expected in problem_cases:
        # Linear prediction
        linear_pred = linear_model.predict([[days, miles, receipts]])[0]
        
        # Interaction prediction
        days_miles = days * miles
        days_receipts = days * receipts
        miles_receipts = miles * receipts
        interaction_pred = interaction_model.predict([[days, miles, receipts, days_miles, days_receipts, miles_receipts]])[0]
        
        # Polynomial prediction
        poly_features = poly_transformer.transform([[days, miles, receipts]])
        poly_pred = poly_model.predict(poly_features)[0]
        
        print(f"{days}d, {miles}mi, ${receipts:.2f} (expected: ${expected:.2f})")
        print(f"  Linear: ${linear_pred:.2f} (error: ${abs(linear_pred - expected):.2f})")
        print(f"  Interaction: ${interaction_pred:.2f} (error: ${abs(interaction_pred - expected):.2f})")
        print(f"  Polynomial: ${poly_pred:.2f} (error: ${abs(poly_pred - expected):.2f})")
        print()

def analyze_specific_patterns(df):
    """Look for specific patterns mentioned in interviews"""
    print("\n\nAnalyzing specific patterns...")
    print("-" * 60)
    
    # 5-day trips
    five_day = df[df['days'] == 5]
    other_days = df[df['days'].isin([4, 6])]
    
    print("5-day trip analysis:")
    print(f"  5-day average: ${five_day['expected'].mean():.2f}")
    print(f"  4,6-day average: ${other_days['expected'].mean():.2f}")
    
    # Rounding bug
    df['ends_49'] = df['receipts'].apply(lambda x: str(x).endswith('.49'))
    df['ends_99'] = df['receipts'].apply(lambda x: str(x).endswith('.99'))
    df['has_rounding'] = df['ends_49'] | df['ends_99']
    
    rounding = df[df['has_rounding']]
    normal = df[~df['has_rounding']]
    
    print(f"\nRounding bug analysis:")
    print(f"  With .49/.99: ${rounding['expected'].mean():.2f} (n={len(rounding)})")
    print(f"  Normal: ${normal['expected'].mean():.2f} (n={len(normal)})")
    
    # Efficiency analysis
    df['efficiency'] = df['miles_per_day']
    sweet_spot = df[(df['efficiency'] >= 180) & (df['efficiency'] <= 220)]
    print(f"\nEfficiency sweet spot (180-220 mi/day):")
    print(f"  Cases: {len(sweet_spot)}")
    print(f"  Average reimbursement: ${sweet_spot['expected'].mean():.2f}")

if __name__ == "__main__":
    df = load_data()
    
    # Basic models
    print("Building models...")
    
    # Simple linear
    X_linear = df[['days', 'miles', 'receipts']].values
    y = df['expected'].values
    linear_model = LinearRegression().fit(X_linear, y)
    
    # With interactions
    interaction_model, features = analyze_interaction_effects(df)
    
    # Polynomial
    poly_model, poly_transformer = analyze_polynomial_effects(df)
    
    # Analyze patterns
    analyze_long_trips(df)
    analyze_specific_patterns(df)
    
    # Test models
    models = (linear_model, interaction_model, poly_model, poly_transformer)
    test_models_on_problem_cases(df, models) 