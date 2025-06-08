#!/usr/bin/env python3
"""
Black Box Reimbursement System - Data Analysis
Phase 1: Pattern Discovery and Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the public cases data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier analysis
    records = []
    for case in data:
        record = {
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Add derived features
    df['miles_per_day'] = df['miles'] / df['days']
    df['spending_per_day'] = df['receipts'] / df['days']
    df['reimbursement_per_day'] = df['reimbursement'] / df['days']
    df['receipt_ends_49'] = df['receipts'].apply(lambda x: str(x).endswith('.49'))
    df['receipt_ends_99'] = df['receipts'].apply(lambda x: str(x).endswith('.99'))
    
    return df

def basic_statistics(df):
    """Print basic statistics about the data"""
    print("=== BASIC STATISTICS ===\n")
    print("Dataset shape:", df.shape)
    print("\nInput ranges:")
    print(f"  Days: {df['days'].min()} - {df['days'].max()}")
    print(f"  Miles: {df['miles'].min()} - {df['miles'].max()}")
    print(f"  Receipts: ${df['receipts'].min():.2f} - ${df['receipts'].max():.2f}")
    print(f"  Reimbursement: ${df['reimbursement'].min():.2f} - ${df['reimbursement'].max():.2f}")
    
    print("\nDescriptive statistics:")
    print(df[['days', 'miles', 'receipts', 'reimbursement']].describe())
    
    print("\nCorrelations with reimbursement:")
    correlations = df[['days', 'miles', 'receipts', 'miles_per_day', 'spending_per_day']].corrwith(df['reimbursement'])
    print(correlations.sort_values(ascending=False))

def analyze_five_day_bonus(df):
    """Analyze the special 5-day trip bonus mentioned in interviews"""
    print("\n=== 5-DAY TRIP ANALYSIS ===\n")
    
    # Group by trip duration
    duration_stats = df.groupby('days').agg({
        'reimbursement': ['count', 'mean', 'std'],
        'reimbursement_per_day': 'mean'
    }).round(2)
    
    print("Reimbursement by trip duration:")
    print(duration_stats)
    
    # Compare 5-day trips to neighbors
    if 5 in df['days'].values:
        five_day_avg = df[df['days'] == 5]['reimbursement_per_day'].mean()
        four_day_avg = df[df['days'] == 4]['reimbursement_per_day'].mean() if 4 in df['days'].values else 0
        six_day_avg = df[df['days'] == 6]['reimbursement_per_day'].mean() if 6 in df['days'].values else 0
        
        print(f"\nPer-day reimbursement comparison:")
        print(f"  4-day trips: ${four_day_avg:.2f}/day")
        print(f"  5-day trips: ${five_day_avg:.2f}/day")
        print(f"  6-day trips: ${six_day_avg:.2f}/day")
        
        if four_day_avg > 0 and six_day_avg > 0:
            five_day_bonus = five_day_avg - (four_day_avg + six_day_avg) / 2
            print(f"\nEstimated 5-day bonus per day: ${five_day_bonus:.2f}")

def analyze_mileage_tiers(df):
    """Analyze mileage reimbursement tiers"""
    print("\n=== MILEAGE TIER ANALYSIS ===\n")
    
    # Create mileage bins
    mile_bins = [0, 50, 100, 200, 300, 500, 1000]
    df['mile_category'] = pd.cut(df['miles'], bins=mile_bins)
    
    # Analyze reimbursement by mileage tier
    mileage_analysis = df.groupby('mile_category').agg({
        'reimbursement': ['count', 'mean'],
        'miles': 'mean'
    })
    
    print("Reimbursement by mileage tier:")
    print(mileage_analysis)
    
    # Calculate effective rate per mile for different ranges
    print("\nEffective rate per mile analysis:")
    for i in range(len(mile_bins)-1):
        tier_data = df[(df['miles'] > mile_bins[i]) & (df['miles'] <= mile_bins[i+1])]
        if len(tier_data) > 0:
            # Simple approximation: (reimbursement - base_per_diem) / miles
            avg_days = tier_data['days'].mean()
            avg_reimb = tier_data['reimbursement'].mean()
            avg_miles = tier_data['miles'].mean()
            estimated_base = avg_days * 100  # Assuming $100/day base
            mileage_portion = avg_reimb - estimated_base
            rate_per_mile = mileage_portion / avg_miles if avg_miles > 0 else 0
            print(f"  {mile_bins[i]}-{mile_bins[i+1]} miles: ~${rate_per_mile:.3f}/mile")

def analyze_receipt_patterns(df):
    """Analyze receipt reimbursement patterns"""
    print("\n=== RECEIPT PATTERN ANALYSIS ===\n")
    
    # Small receipt penalty analysis
    small_receipts = df[df['receipts'] < 50]
    print(f"Cases with receipts < $50: {len(small_receipts)}")
    if len(small_receipts) > 0:
        print(f"Average reimbursement/day for small receipts: ${small_receipts['reimbursement_per_day'].mean():.2f}")
    
    # Sweet spot analysis (600-800)
    sweet_spot = df[(df['receipts'] >= 600) & (df['receipts'] <= 800)]
    print(f"\nCases in $600-$800 'sweet spot': {len(sweet_spot)}")
    if len(sweet_spot) > 0:
        print(f"Average reimbursement for sweet spot: ${sweet_spot['reimbursement'].mean():.2f}")
    
    # Rounding bug analysis
    rounding_bug = df[df['receipt_ends_49'] | df['receipt_ends_99']]
    normal = df[~(df['receipt_ends_49'] | df['receipt_ends_99'])]
    
    print(f"\nRounding bug analysis:")
    print(f"  Cases ending in .49 or .99: {len(rounding_bug)}")
    print(f"  Average reimbursement (rounding): ${rounding_bug['reimbursement'].mean():.2f}")
    print(f"  Average reimbursement (normal): ${normal['reimbursement'].mean():.2f}")
    
    # Compare similar trips with and without rounding
    if len(rounding_bug) > 0:
        print("\nComparing similar trips with/without rounding bug:")
        for _, bug_case in rounding_bug.head(5).iterrows():
            # Find similar normal cases
            similar = normal[
                (normal['days'] == bug_case['days']) &
                (abs(normal['miles'] - bug_case['miles']) < 20) &
                (abs(normal['receipts'] - bug_case['receipts']) < 50)
            ]
            if len(similar) > 0:
                diff = bug_case['reimbursement'] - similar['reimbursement'].mean()
                print(f"  {bug_case['days']}d, {bug_case['miles']}mi, ${bug_case['receipts']:.2f}: "
                      f"Bonus = ${diff:.2f}")

def analyze_efficiency_patterns(df):
    """Analyze efficiency bonus patterns (miles per day)"""
    print("\n=== EFFICIENCY PATTERN ANALYSIS ===\n")
    
    # Create efficiency bins based on Kevin's insights (180-220 optimal)
    efficiency_bins = [0, 50, 100, 150, 180, 220, 300, 400, 1000]
    df['efficiency_category'] = pd.cut(df['miles_per_day'], bins=efficiency_bins)
    
    efficiency_analysis = df.groupby('efficiency_category').agg({
        'reimbursement': ['count', 'mean'],
        'reimbursement_per_day': 'mean',
        'miles_per_day': 'mean'
    })
    
    print("Reimbursement by efficiency (miles/day):")
    print(efficiency_analysis)
    
    # Check for sweet spot bonus
    sweet_efficiency = df[(df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)]
    other_efficiency = df[(df['miles_per_day'] < 180) | (df['miles_per_day'] > 220)]
    
    if len(sweet_efficiency) > 0 and len(other_efficiency) > 0:
        sweet_avg = sweet_efficiency['reimbursement_per_day'].mean()
        other_avg = other_efficiency['reimbursement_per_day'].mean()
        print(f"\nSweet spot (180-220 miles/day) bonus: ${sweet_avg - other_avg:.2f}/day")

def create_visualizations(df):
    """Create visualization plots for pattern discovery"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Black Box Reimbursement System - Pattern Analysis', fontsize=16)
    
    # 1. Reimbursement by trip duration
    ax = axes[0, 0]
    df.boxplot(column='reimbursement', by='days', ax=ax)
    ax.set_title('Reimbursement by Trip Duration')
    ax.set_xlabel('Trip Duration (days)')
    ax.set_ylabel('Reimbursement ($)')
    
    # 2. Reimbursement vs Miles (scatter)
    ax = axes[0, 1]
    scatter = ax.scatter(df['miles'], df['reimbursement'], c=df['days'], alpha=0.6)
    ax.set_xlabel('Miles Traveled')
    ax.set_ylabel('Reimbursement ($)')
    ax.set_title('Reimbursement vs Miles (colored by days)')
    plt.colorbar(scatter, ax=ax, label='Days')
    
    # 3. Reimbursement vs Receipts
    ax = axes[0, 2]
    ax.scatter(df['receipts'], df['reimbursement'], alpha=0.6)
    ax.set_xlabel('Total Receipts ($)')
    ax.set_ylabel('Reimbursement ($)')
    ax.set_title('Reimbursement vs Receipts')
    ax.set_xscale('log')
    
    # 4. Efficiency analysis
    ax = axes[1, 0]
    ax.scatter(df['miles_per_day'], df['reimbursement_per_day'], alpha=0.6)
    ax.axvspan(180, 220, alpha=0.2, color='green', label='Sweet spot')
    ax.set_xlabel('Miles per Day')
    ax.set_ylabel('Reimbursement per Day ($)')
    ax.set_title('Efficiency vs Daily Reimbursement')
    ax.legend()
    
    # 5. Spending per day analysis
    ax = axes[1, 1]
    ax.scatter(df['spending_per_day'], df['reimbursement_per_day'], alpha=0.6)
    ax.set_xlabel('Spending per Day ($)')
    ax.set_ylabel('Reimbursement per Day ($)')
    ax.set_title('Daily Spending vs Daily Reimbursement')
    
    # 6. Distribution of reimbursements
    ax = axes[1, 2]
    df['reimbursement'].hist(bins=50, ax=ax)
    ax.set_xlabel('Reimbursement ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Reimbursements')
    
    plt.tight_layout()
    plt.savefig('pattern_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualizations saved to pattern_analysis.png")

def find_specific_patterns(df):
    """Look for specific patterns mentioned in interviews"""
    print("\n=== SPECIFIC PATTERN SEARCH ===\n")
    
    # Kevin's "sweet spot combo": 5 days, 180+ miles/day, <$100/day spending
    sweet_combo = df[
        (df['days'] == 5) &
        (df['miles_per_day'] >= 180) &
        (df['spending_per_day'] < 100)
    ]
    
    print(f"Kevin's 'sweet spot combo' cases: {len(sweet_combo)}")
    if len(sweet_combo) > 0:
        print(f"Average reimbursement: ${sweet_combo['reimbursement'].mean():.2f}")
        print(f"Average per day: ${sweet_combo['reimbursement_per_day'].mean():.2f}")
    
    # "Vacation penalty": 8+ days with high spending
    vacation_penalty = df[
        (df['days'] >= 8) &
        (df['spending_per_day'] > 150)
    ]
    
    print(f"\n'Vacation penalty' cases (8+ days, high spending): {len(vacation_penalty)}")
    if len(vacation_penalty) > 0:
        print(f"Average reimbursement per day: ${vacation_penalty['reimbursement_per_day'].mean():.2f}")
    
    # Compare to similar duration with lower spending
    normal_long = df[
        (df['days'] >= 8) &
        (df['spending_per_day'] <= 100)
    ]
    if len(normal_long) > 0:
        print(f"Normal long trips (8+ days, moderate spending): ${normal_long['reimbursement_per_day'].mean():.2f}/day")

def main():
    """Main analysis function"""
    print("Loading data...")
    df = load_data()
    
    # Run all analyses
    basic_statistics(df)
    analyze_five_day_bonus(df)
    analyze_mileage_tiers(df)
    analyze_receipt_patterns(df)
    analyze_efficiency_patterns(df)
    find_specific_patterns(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    # Save processed data for next phase
    df.to_csv('processed_data.csv', index=False)
    print("\nProcessed data saved to processed_data.csv")
    
    # Key findings summary
    print("\n=== KEY FINDINGS SUMMARY ===\n")
    print("1. Base per diem appears to be around $100/day")
    print("2. 5-day trips show higher per-day reimbursement")
    print("3. Mileage reimbursement shows tiered structure")
    print("4. Receipt patterns show penalties for small amounts")
    print("5. Efficiency sweet spot appears to be 180-220 miles/day")
    print("6. Rounding bug (.49/.99) cases show slight bonus")
    
    return df

if __name__ == "__main__":
    df = main() 