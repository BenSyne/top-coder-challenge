#!/usr/bin/env python3
"""
Analyze high receipt cases to understand the pattern
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def analyze_high_receipts(df):
    """Analyze cases with high receipts"""
    print("HIGH RECEIPT ANALYSIS")
    print("=" * 60)
    
    # Look at cases with receipts > 2000
    high_receipts = df[df['receipts'] > 2000].sort_values('receipts')
    
    print(f"Found {len(high_receipts)} cases with receipts > $2000\n")
    
    # Calculate what simple formulas would predict
    for idx, row in high_receipts.head(20).iterrows():
        # Simple formula prediction
        simple = 75 * row['days'] + 0.5 * row['miles'] + 0.4 * row['receipts']
        
        # What fraction of receipts is reimbursed?
        receipt_fraction = (row['expected'] - 75 * row['days'] - 0.5 * row['miles']) / row['receipts']
        
        print(f"Days: {row['days']:2.0f}, Miles: {row['miles']:4.0f}, Receipts: ${row['receipts']:7.2f}")
        print(f"  Expected: ${row['expected']:7.2f}")
        print(f"  Simple formula: ${simple:7.2f} (error: ${simple - row['expected']:7.2f})")
        print(f"  Implied receipt rate: {receipt_fraction:.3f}")
        print()

def analyze_receipt_thresholds(df):
    """Look for thresholds in receipt processing"""
    print("\nRECEIPT THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Group by receipt ranges and calculate average reimbursement rates
    ranges = [
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000),
        (2000, 2500),
        (2500, 3000)
    ]
    
    for min_r, max_r in ranges:
        subset = df[(df['receipts'] >= min_r) & (df['receipts'] < max_r)]
        if len(subset) > 0:
            # Calculate average reimbursement per dollar of receipts
            # After removing estimated day and mile components
            subset['net_receipts'] = subset['expected'] - 75 * subset['days'] - 0.5 * subset['miles']
            subset['receipt_rate'] = subset['net_receipts'] / subset['receipts']
            
            avg_rate = subset['receipt_rate'].mean()
            std_rate = subset['receipt_rate'].std()
            
            print(f"${min_r:4d}-${max_r:4d}: {len(subset):3d} cases, "
                  f"avg rate: {avg_rate:6.3f} (std: {std_rate:5.3f})")

def test_capped_receipts(df):
    """Test if receipts might be capped at certain amounts"""
    print("\n\nTESTING RECEIPT CAP HYPOTHESIS")
    print("=" * 60)
    
    # Test different cap amounts
    caps = [1000, 1200, 1500, 1800, 2000]
    
    for cap in caps:
        total_error = 0
        count = 0
        
        for _, row in df.iterrows():
            # Calculate with capped receipts
            capped_receipts = min(row['receipts'], cap)
            predicted = 75 * row['days'] + 0.5 * row['miles'] + 0.4 * capped_receipts
            
            error = abs(predicted - row['expected'])
            total_error += error
            count += 1
        
        avg_error = total_error / count
        print(f"Receipt cap at ${cap}: avg error ${avg_error:.2f}")

def plot_receipt_analysis(df):
    """Create visualizations for receipt patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Receipt Pattern Analysis', fontsize=16)
    
    # 1. Expected vs Receipts
    ax = axes[0, 0]
    ax.scatter(df['receipts'], df['expected'], alpha=0.5, s=10)
    ax.set_xlabel('Total Receipts ($)')
    ax.set_ylabel('Expected Reimbursement ($)')
    ax.set_title('Reimbursement vs Receipts')
    
    # Add trend lines for different ranges
    for max_r in [1000, 2000, 3000]:
        subset = df[df['receipts'] <= max_r]
        if len(subset) > 10:
            z = np.polyfit(subset['receipts'], subset['expected'], 1)
            p = np.poly1d(z)
            ax.plot([0, max_r], [p(0), p(max_r)], '--', alpha=0.7, 
                   label=f'≤${max_r}')
    ax.legend()
    
    # 2. Receipt processing rate by amount
    ax = axes[0, 1]
    df['receipt_rate'] = (df['expected'] - 75 * df['days'] - 0.5 * df['miles']) / df['receipts']
    
    # Group into bins
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000]
    df['receipt_bin'] = pd.cut(df['receipts'], bins)
    grouped = df.groupby('receipt_bin')['receipt_rate'].mean()
    
    x_pos = range(len(grouped))
    ax.bar(x_pos, grouped.values)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"${int(b.left)}-{int(b.right)}" for b in grouped.index], rotation=45)
    ax.set_ylabel('Average Receipt Rate')
    ax.set_title('Receipt Processing Rate by Amount')
    
    # 3. High receipt cases
    ax = axes[1, 0]
    high = df[df['receipts'] > 1500]
    if len(high) > 0:
        ax.scatter(high['receipts'], high['expected'], alpha=0.7, s=30, c='red')
        ax.set_xlabel('Receipts ($)')
        ax.set_ylabel('Expected ($)')
        ax.set_title('High Receipt Cases (>$1500)')
    
    # 4. Error pattern for high receipts
    ax = axes[1, 1]
    # Calculate simple prediction error
    df['simple_pred'] = 75 * df['days'] + 0.5 * df['miles'] + 0.4 * df['receipts']
    df['error'] = df['simple_pred'] - df['expected']
    
    ax.scatter(df['receipts'], df['error'], alpha=0.5, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Receipts ($)')
    ax.set_ylabel('Prediction Error ($)')
    ax.set_title('Error vs Receipt Amount')
    
    plt.tight_layout()
    plt.savefig('receipt_analysis.png', dpi=300)
    print("\nVisualizations saved to receipt_analysis.png")

if __name__ == "__main__":
    df = load_data()
    
    analyze_high_receipts(df)
    analyze_receipt_thresholds(df)
    test_capped_receipts(df)
    plot_receipt_analysis(df) 