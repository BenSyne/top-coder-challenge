#!/usr/bin/env python3
"""
Detailed error analysis to understand prediction patterns
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_reimbursement import ReimbursementCalculator

def load_and_predict():
    """Load data and generate predictions"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calculator = ReimbursementCalculator()
    results = []
    
    for case in data:
        days = case['input']['trip_duration_days']
        miles = int(case['input']['miles_traveled'])
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = predicted - expected
        percent_error = (error / expected) * 100 if expected > 0 else 0
        
        results.append({
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'expected': expected,
            'predicted': predicted,
            'error': error,
            'abs_error': abs(error),
            'percent_error': percent_error,
            'miles_per_day': miles / days,
            'receipts_per_day': receipts / days,
            'rounding_bug': str(receipts).endswith('.49') or str(receipts).endswith('.99')
        })
    
    return pd.DataFrame(results)

def analyze_error_distribution(df):
    """Analyze the distribution of errors"""
    print("ERROR DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    print(f"Total cases: {len(df)}")
    print(f"Mean error: ${df['error'].mean():.2f}")
    print(f"Median error: ${df['error'].median():.2f}")
    print(f"Std deviation: ${df['error'].std():.2f}")
    
    print(f"\nExact matches (±$0.01): {len(df[df['abs_error'] <= 0.01])}")
    print(f"Close matches (±$1.00): {len(df[df['abs_error'] <= 1.00])}")
    print(f"Within ±$10: {len(df[df['abs_error'] <= 10])}")
    print(f"Within ±$50: {len(df[df['abs_error'] <= 50])}")
    
    print("\nError percentiles:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(df['abs_error'], p)
        print(f"  {p}th percentile: ${value:.2f}")

def analyze_best_matches(df):
    """Analyze cases with smallest errors"""
    print("\n\nBEST MATCHES (smallest absolute errors)")
    print("=" * 60)
    
    best = df.nsmallest(20, 'abs_error')
    
    print(f"{'Days':>4} {'Miles':>6} {'Receipts':>10} {'Expected':>10} {'Predicted':>10} {'Error':>8}")
    print("-" * 60)
    
    for _, row in best.iterrows():
        print(f"{row['days']:4.0f} {row['miles']:6.0f} {row['receipts']:10.2f} "
              f"{row['expected']:10.2f} {row['predicted']:10.2f} {row['error']:8.2f}")
    
    # Look for patterns in best matches
    print("\nPatterns in best matches:")
    print(f"  Average days: {best['days'].mean():.1f}")
    print(f"  Average miles: {best['miles'].mean():.0f}")
    print(f"  Average receipts: ${best['receipts'].mean():.2f}")
    print(f"  Average miles/day: {best['miles_per_day'].mean():.0f}")

def analyze_worst_matches(df):
    """Analyze cases with largest errors"""
    print("\n\nWORST MATCHES (largest absolute errors)")
    print("=" * 60)
    
    worst = df.nlargest(20, 'abs_error')
    
    print(f"{'Days':>4} {'Miles':>6} {'Receipts':>10} {'Expected':>10} {'Predicted':>10} {'Error':>8}")
    print("-" * 60)
    
    for _, row in worst.iterrows():
        print(f"{row['days']:4.0f} {row['miles']:6.0f} {row['receipts']:10.2f} "
              f"{row['expected']:10.2f} {row['predicted']:10.2f} {row['error']:8.2f}")
    
    print("\nPatterns in worst matches:")
    print(f"  Average days: {worst['days'].mean():.1f}")
    print(f"  Average miles: {worst['miles'].mean():.0f}")
    print(f"  Average receipts: ${worst['receipts'].mean():.2f}")
    print(f"  Average miles/day: {worst['miles_per_day'].mean():.0f}")

def analyze_by_category(df):
    """Analyze errors by trip category"""
    print("\n\nERROR ANALYSIS BY CATEGORY")
    print("=" * 60)
    
    # By trip duration
    print("By trip duration:")
    for days in sorted(df['days'].unique()):
        subset = df[df['days'] == days]
        print(f"  {days} days: mean error ${subset['error'].mean():6.2f}, "
              f"median ${subset['error'].median():6.2f} (n={len(subset)})")
    
    # By mileage ranges
    print("\nBy mileage range:")
    mile_ranges = [(0, 100), (100, 300), (300, 600), (600, 900), (900, 1500)]
    for min_m, max_m in mile_ranges:
        subset = df[(df['miles'] >= min_m) & (df['miles'] < max_m)]
        if len(subset) > 0:
            print(f"  {min_m:4d}-{max_m:4d} miles: mean error ${subset['error'].mean():6.2f}, "
                  f"median ${subset['error'].median():6.2f} (n={len(subset)})")
    
    # Rounding bug cases
    print("\nRounding bug analysis:")
    rounding = df[df['rounding_bug']]
    normal = df[~df['rounding_bug']]
    print(f"  With .49/.99: mean error ${rounding['error'].mean():.2f} (n={len(rounding)})")
    print(f"  Normal: mean error ${normal['error'].mean():.2f} (n={len(normal)})")

def find_systematic_patterns(df):
    """Look for systematic patterns in errors"""
    print("\n\nSYSTEMATIC PATTERNS")
    print("=" * 60)
    
    # Check if we're consistently over or under
    over_predictions = len(df[df['error'] > 0])
    under_predictions = len(df[df['error'] < 0])
    
    print(f"Over-predictions: {over_predictions} ({over_predictions/len(df)*100:.1f}%)")
    print(f"Under-predictions: {under_predictions} ({under_predictions/len(df)*100:.1f}%)")
    
    # Check correlation between error and inputs
    print("\nCorrelation between error and inputs:")
    correlations = df[['days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day']].corrwith(df['error'])
    for col, corr in correlations.items():
        print(f"  {col}: {corr:.3f}")
    
    # Look for threshold effects
    print("\nPotential threshold effects:")
    # Check if errors change dramatically at certain values
    for threshold in [100, 200, 500, 1000]:
        below = df[df['miles'] < threshold]
        above = df[df['miles'] >= threshold]
        if len(below) > 10 and len(above) > 10:
            diff = above['error'].mean() - below['error'].mean()
            print(f"  Miles {threshold}: error diff = ${diff:.2f}")

def create_error_visualizations(df):
    """Create visualizations of error patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Error Analysis Visualizations', fontsize=16)
    
    # 1. Error distribution histogram
    ax = axes[0, 0]
    ax.hist(df['error'], bins=50, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Zero error')
    ax.set_xlabel('Prediction Error ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # 2. Error vs Expected
    ax = axes[0, 1]
    ax.scatter(df['expected'], df['error'], alpha=0.5, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Expected Reimbursement ($)')
    ax.set_ylabel('Prediction Error ($)')
    ax.set_title('Error vs Expected Value')
    
    # 3. Error by trip duration
    ax = axes[1, 0]
    df.boxplot(column='error', by='days', ax=ax)
    ax.set_xlabel('Trip Duration (days)')
    ax.set_ylabel('Prediction Error ($)')
    ax.set_title('Error by Trip Duration')
    
    # 4. Absolute error vs miles
    ax = axes[1, 1]
    ax.scatter(df['miles'], df['abs_error'], alpha=0.5, s=10)
    ax.set_xlabel('Miles Traveled')
    ax.set_ylabel('Absolute Error ($)')
    ax.set_title('Absolute Error vs Miles')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300)
    print("\nVisualizations saved to error_analysis.png")

def suggest_improvements(df):
    """Suggest specific improvements based on error patterns"""
    print("\n\nSUGGESTED IMPROVEMENTS")
    print("=" * 60)
    
    # Check if long trips are systematically underestimated
    long_trips = df[df['days'] >= 10]
    if len(long_trips) > 0 and long_trips['error'].mean() < -100:
        print("1. Long trips (10+ days) are underestimated by ${:.2f} on average".format(
            abs(long_trips['error'].mean())))
        print("   → Consider increasing the multiplier for long trips")
    
    # Check high mileage trips
    high_mileage = df[df['miles'] >= 1000]
    if len(high_mileage) > 0:
        print(f"\n2. High mileage trips (1000+ miles) have mean error: ${high_mileage['error'].mean():.2f}")
        if abs(high_mileage['error'].mean()) > 200:
            print("   → Need special handling for very high mileage")
    
    # Check rounding bug adjustment
    rounding = df[df['rounding_bug']]
    if len(rounding) > 0:
        print(f"\n3. Rounding bug cases have mean error: ${rounding['error'].mean():.2f}")
        print(f"   Current factor: 0.41")
        
        # Calculate what the factor should be
        avg_predicted = rounding['predicted'].mean()
        avg_expected = rounding['expected'].mean()
        if avg_predicted > 0:
            suggested_factor = avg_expected / (avg_predicted / 0.41)
            print(f"   → Suggested factor: {suggested_factor:.3f}")

if __name__ == "__main__":
    print("Loading data and generating predictions...")
    df = load_and_predict()
    
    analyze_error_distribution(df)
    analyze_best_matches(df)
    analyze_worst_matches(df)
    analyze_by_category(df)
    find_systematic_patterns(df)
    suggest_improvements(df)
    create_error_visualizations(df)
    
    # Save detailed results for manual inspection
    df.to_csv('error_analysis_results.csv', index=False)
    print("\nDetailed results saved to error_analysis_results.csv") 