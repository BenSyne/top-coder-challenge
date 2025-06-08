#!/usr/bin/env python3
"""
Deep analysis of Ultra ML Calculator errors to find remaining patterns
Goal: Find systematic issues to push score towards zero
"""

import json
import numpy as np
from ultra_ml_calculator import UltraMLCalculator
import matplotlib.pyplot as plt
from collections import defaultdict

def deep_error_analysis():
    """Comprehensive analysis of remaining Ultra ML errors"""
    
    print("🔬 DEEP ANALYSIS OF ULTRA ML ERRORS")
    print("=" * 70)
    print("Goal: Identify remaining patterns to achieve near-zero error")
    print()
    
    # Load test data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Initialize calculator
    print("Loading Ultra ML model...")
    calculator = UltraMLCalculator()
    
    # Collect detailed error information
    errors = []
    error_by_exact_amount = defaultdict(list)
    error_by_pattern = defaultdict(list)
    
    print("Analyzing errors...")
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = predicted - expected  # Signed error
        abs_error = abs(error)
        pct_error = abs_error / expected * 100 if expected > 0 else 0
        
        error_info = {
            'index': i,
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'expected': expected,
            'predicted': predicted,
            'error': error,
            'abs_error': abs_error,
            'pct_error': pct_error,
            'miles_per_day': miles / days if days > 0 else miles,
            'receipts_per_day': receipts / days if days > 0 else receipts,
            'receipts_per_mile': receipts / miles if miles > 0 else 0
        }
        
        errors.append(error_info)
        
        # Group by exact expected amounts (might reveal formula patterns)
        error_by_exact_amount[expected].append(error_info)
        
        # Categorize by various patterns
        receipt_str = f"{receipts:.2f}"
        if receipt_str.endswith('.49') or receipt_str.endswith('.99'):
            error_by_pattern['rounding_bug'].append(error_info)
        
        if abs_error < 1.0:
            error_by_pattern['almost_perfect'].append(error_info)
        elif abs_error > 20:
            error_by_pattern['large_error'].append(error_info)
        
        # Edge cases
        if days == 1:
            error_by_pattern['single_day'].append(error_info)
        if days == 30:
            error_by_pattern['max_days'].append(error_info)
        if miles == 0:
            error_by_pattern['zero_miles'].append(error_info)
        if receipts == 0:
            error_by_pattern['zero_receipts'].append(error_info)
        
        # Specific problematic ranges from previous analysis
        if 8 <= days <= 14 and 600 <= miles <= 1100:
            error_by_pattern['problem_range_1'].append(error_info)
        if days == 7 and 1000 <= receipts <= 2000:
            error_by_pattern['problem_range_2'].append(error_info)
    
    # Sort errors
    errors_sorted = sorted(errors, key=lambda x: x['abs_error'], reverse=True)
    
    # Basic statistics
    all_errors = [e['error'] for e in errors]
    abs_errors = [e['abs_error'] for e in errors]
    
    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 50)
    print(f"Average error: ${np.mean(abs_errors):.2f}")
    print(f"Median error: ${np.median(abs_errors):.2f}")
    print(f"Std deviation: ${np.std(abs_errors):.2f}")
    print(f"Max error: ${max(abs_errors):.2f}")
    print(f"Min error: ${min(abs_errors):.2f}")
    
    # Error distribution
    print(f"\n📈 ERROR DISTRIBUTION")
    print("-" * 50)
    error_bins = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    for i in range(len(error_bins)-1):
        count = sum(1 for e in abs_errors if error_bins[i] <= e < error_bins[i+1])
        pct = count / len(abs_errors) * 100
        print(f"${error_bins[i]:5.1f} - ${error_bins[i+1]:5.1f}: {count:4d} cases ({pct:5.1f}%)")
    
    # Perfect and near-perfect predictions
    perfect = sum(1 for e in abs_errors if e < 0.01)
    near_perfect = sum(1 for e in abs_errors if e < 1.0)
    print(f"\nPerfect predictions (<$0.01): {perfect} ({perfect/len(errors)*100:.1f}%)")
    print(f"Near-perfect (<$1.00): {near_perfect} ({near_perfect/len(errors)*100:.1f}%)")
    
    # Analyze worst cases
    print(f"\n❌ TOP 15 WORST PREDICTIONS")
    print("-" * 80)
    print("Idx  Days Miles Receipts  MPD   RPD   Expected  Predicted   Error  %Err")
    print("-" * 80)
    for e in errors_sorted[:15]:
        print(f"{e['index']:3d}  {e['days']:4d} {e['miles']:5.0f} {e['receipts']:8.2f} "
              f"{e['miles_per_day']:5.0f} {e['receipts_per_day']:6.0f} "
              f"{e['expected']:9.2f} {e['predicted']:9.2f} {e['error']:7.2f} {e['pct_error']:5.1f}%")
    
    # Find systematic biases
    print(f"\n🎯 SYSTEMATIC BIAS ANALYSIS")
    print("-" * 50)
    
    # By day count
    print("\nBias by trip length:")
    for day in range(1, 16):
        day_cases = [e for e in errors if e['days'] == day]
        if len(day_cases) >= 5:
            avg_error = np.mean([e['error'] for e in day_cases])
            avg_abs = np.mean([e['abs_error'] for e in day_cases])
            if abs(avg_error) > 1 or avg_abs > 5:
                print(f"  Day {day:2d}: {len(day_cases):3d} cases, "
                      f"bias=${avg_error:6.2f}, avg|error|=${avg_abs:5.2f}")
    
    # Pattern analysis
    print(f"\n🔍 PATTERN-SPECIFIC ERRORS")
    print("-" * 50)
    for pattern, cases in error_by_pattern.items():
        if cases and pattern not in ['almost_perfect']:
            avg_abs = np.mean([c['abs_error'] for c in cases])
            if avg_abs > 1:
                print(f"{pattern:15}: {len(cases):4d} cases, ${avg_abs:6.2f} avg error")
    
    # Find cases with similar expected values but different errors
    print(f"\n💡 FORMULA CLUES (similar expected, different errors)")
    print("-" * 70)
    
    # Group by rounded expected values
    expected_groups = defaultdict(list)
    for e in errors:
        rounded_expected = round(e['expected'], 0)
        expected_groups[rounded_expected].append(e)
    
    # Find groups with high variance in predictions
    interesting_groups = []
    for expected, group in expected_groups.items():
        if len(group) >= 3:
            predictions = [g['predicted'] for g in group]
            pred_std = np.std(predictions)
            if pred_std > 5:  # High variance in predictions
                interesting_groups.append((expected, group, pred_std))
    
    # Show top interesting groups
    for expected, group, std in sorted(interesting_groups, key=lambda x: x[2], reverse=True)[:5]:
        print(f"\nExpected ~${expected:.0f} (std=${std:.1f}):")
        for g in sorted(group, key=lambda x: x['abs_error'], reverse=True)[:3]:
            print(f"  Case {g['index']:3d}: {g['days']:2d}d, {g['miles']:4.0f}mi, "
                  f"${g['receipts']:7.2f} → ${g['predicted']:7.2f} (err ${g['abs_error']:5.2f})")
    
    return errors_sorted, error_by_pattern

def find_micro_patterns(errors_sorted):
    """Look for very specific patterns in remaining errors"""
    
    print(f"\n🔬 MICRO-PATTERN DETECTION")
    print("=" * 70)
    
    # Analyze errors > $10
    significant_errors = [e for e in errors_sorted if e['abs_error'] > 10]
    
    if significant_errors:
        print(f"\nAnalyzing {len(significant_errors)} cases with error > $10:")
        
        # Check for decimal patterns in inputs
        decimal_patterns = defaultdict(list)
        for e in significant_errors:
            # Check receipt decimals
            receipt_cents = int(round((e['receipts'] % 1) * 100))
            decimal_patterns[receipt_cents].append(e)
        
        print("\nReceipt decimal patterns in high-error cases:")
        for cents, cases in sorted(decimal_patterns.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            if len(cases) >= 2:
                avg_error = np.mean([c['abs_error'] for c in cases])
                print(f"  .{cents:02d}: {len(cases)} cases, ${avg_error:.2f} avg error")
        
        # Check for ratio patterns
        print("\nRatio patterns in high-error cases:")
        for e in significant_errors[:10]:
            ratio1 = e['receipts'] / e['miles'] if e['miles'] > 0 else 999
            ratio2 = e['receipts'] / e['days'] if e['days'] > 0 else 999
            ratio3 = e['miles'] / e['days'] if e['days'] > 0 else 999
            print(f"  Case {e['index']:3d}: R/M={ratio1:6.2f}, R/D={ratio2:6.2f}, "
                  f"M/D={ratio3:6.2f}, err=${e['abs_error']:5.2f}")
    
    # Check for clustering of errors
    print(f"\n📍 ERROR CLUSTERING")
    print("-" * 50)
    
    # Sort by days, then miles, then receipts
    sorted_by_inputs = sorted(errors_sorted, key=lambda x: (x['days'], x['miles'], x['receipts']))
    
    # Find consecutive cases with similar errors
    clusters = []
    current_cluster = [sorted_by_inputs[0]]
    
    for i in range(1, len(sorted_by_inputs)):
        prev = sorted_by_inputs[i-1]
        curr = sorted_by_inputs[i]
        
        # Check if similar inputs
        if (abs(prev['days'] - curr['days']) <= 1 and 
            abs(prev['miles'] - curr['miles']) <= 100 and
            abs(prev['receipts'] - curr['receipts']) <= 200):
            current_cluster.append(curr)
        else:
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            current_cluster = [curr]
    
    if len(current_cluster) >= 3:
        clusters.append(current_cluster)
    
    print(f"Found {len(clusters)} error clusters")
    for i, cluster in enumerate(clusters[:5]):
        avg_error = np.mean([c['abs_error'] for c in cluster])
        if avg_error > 5:
            print(f"\nCluster {i+1} ({len(cluster)} cases, ${avg_error:.2f} avg error):")
            for c in cluster[:3]:
                print(f"  {c['days']:2d}d, {c['miles']:4.0f}mi, ${c['receipts']:6.2f} "
                      f"→ err ${c['abs_error']:5.2f}")

def suggest_improvements():
    """Suggest specific improvements based on analysis"""
    
    print(f"\n💡 IMPROVEMENT STRATEGIES")
    print("=" * 70)
    
    print("\n1. ENHANCED META-LEARNING:")
    print("   - Add second-level meta-learner trained on first meta-learner's errors")
    print("   - Use isotonic regression for monotonic constraints")
    print("   - Implement confidence-weighted ensemble")
    
    print("\n2. SPECIALIZED MODELS:")
    print("   - Separate model for rounding bug cases (.49/.99)")
    print("   - Dedicated model for single-day trips")
    print("   - Special handling for zero miles/receipts")
    
    print("\n3. FEATURE ENGINEERING:")
    print("   - Add decimal-specific features (last 2 digits of receipts)")
    print("   - Include modulo features (receipts % 10, % 50, % 100)")
    print("   - Add more complex ratios and their logs")
    
    print("\n4. ERROR CORRECTION NETWORK:")
    print("   - Train a neural network specifically on (prediction, actual) pairs")
    print("   - Use gradient boosting on prediction errors")
    print("   - Implement local linear correction models")
    
    print("\n5. ENSEMBLE IMPROVEMENTS:")
    print("   - Add CatBoost and LightGBM models")
    print("   - Implement dynamic weighting based on input characteristics")
    print("   - Use Bayesian model averaging")

def main():
    """Run comprehensive error analysis"""
    errors_sorted, error_by_pattern = deep_error_analysis()
    find_micro_patterns(errors_sorted)
    suggest_improvements()
    
    print("\n🎯 NEXT STEPS:")
    print("=" * 70)
    print("Based on this analysis, the most promising approaches are:")
    print("1. Add a correction network trained on Ultra ML's errors")
    print("2. Implement specialized models for edge cases")
    print("3. Use more advanced ensemble techniques")
    print("4. Add 20-30 more targeted features")
    print("\nEstimated potential: Score < 100 (99%+ accuracy)")

if __name__ == "__main__":
    main() 