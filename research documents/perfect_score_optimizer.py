#!/usr/bin/env python3
"""
Perfect Score Optimizer - Advanced techniques for achieving 1000/1000 exact matches
"""

import json
import math
import sys
from itertools import product
from collections import defaultdict

class PerfectScoreCalculator:
    """Advanced calculator targeting perfect score"""
    
    def __init__(self, config=None):
        # Base coefficients (proven optimal)
        self.intercept = 266.71
        self.coef_days = 50.05
        self.coef_miles = 0.4456
        self.coef_receipts = 0.3829
        self.rounding_bug_factor = 0.457
        
        # Range multipliers (default to current best)
        if config:
            self.range_multipliers = config.get('range_multipliers', {
                (1, 2): 1.05, (3, 4): 1.10, (5, 7): 1.15, (8, 14): 1.00, (15, 30): 1.05
            })
            self.use_nickel_rounding = config.get('use_nickel_rounding', False)
            self.global_adjustment = config.get('global_adjustment', 1.0)
        else:
            self.range_multipliers = {
                (1, 2): 1.05, (3, 4): 1.10, (5, 7): 1.15, (8, 14): 1.00, (15, 30): 1.05
            }
            self.use_nickel_rounding = False
            self.global_adjustment = 1.0
        
    def get_range_multiplier(self, days):
        """Get range multiplier for trip length"""
        for (min_days, max_days), multiplier in self.range_multipliers.items():
            if min_days <= days <= max_days:
                return multiplier
        return 1.00
        
    def calculate(self, days, miles, receipts):
        """Calculate with all optimizations"""
        
        # Check for rounding bug
        receipt_str = f"{receipts:.2f}"
        has_rounding_bug = receipt_str.endswith('.49') or receipt_str.endswith('.99')
        
        # Apply caps
        capped_receipts = receipts
        if receipts > 1800:
            capped_receipts = 1800 + (receipts - 1800) * 0.15
        
        capped_miles = miles
        if miles > 800:
            capped_miles = 800 + (miles - 800) * 0.25
        
        # Base calculation
        amount = (self.intercept + 
                 self.coef_days * days + 
                 self.coef_miles * capped_miles + 
                 self.coef_receipts * capped_receipts)
        
        # Apply rounding bug
        if has_rounding_bug:
            amount *= self.rounding_bug_factor
        else:
            # Legacy adjustments
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
        
        # Apply range multiplier
        if not has_rounding_bug:
            range_multiplier = self.get_range_multiplier(days)
            amount *= range_multiplier
        
        # Apply global adjustment
        amount *= self.global_adjustment
        
        # Apply rounding strategy
        if self.use_nickel_rounding:
            result = round(amount * 20) / 20
        else:
            result = round(amount, 2)
        
        return max(0.0, result)

def fine_tune_range_multipliers():
    """Fine-tune range multipliers for optimal performance"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("🔧 FINE-TUNING RANGE MULTIPLIERS")
    print("=" * 60)
    
    # Current best multipliers
    base_multipliers = {
        (1, 2): 1.05,
        (3, 4): 1.10, 
        (5, 7): 1.15,
        (8, 14): 1.00,
        (15, 30): 1.05
    }
    
    best_score = float('inf')
    best_config = None
    
    # Test fine variations
    variations = {
        (1, 2): [1.04, 1.05, 1.06],
        (3, 4): [1.09, 1.10, 1.11, 1.12],
        (5, 7): [1.14, 1.15, 1.16, 1.17],
        (8, 14): [0.99, 1.00, 1.01],
        (15, 30): [1.04, 1.05, 1.06]
    }
    
    # Test subset of combinations to avoid explosion
    test_count = 0
    max_tests = 200
    
    for mult_1_2 in variations[(1, 2)]:
        for mult_3_4 in variations[(3, 4)]:
            for mult_5_7 in variations[(5, 7)]:
                for mult_8_14 in variations[(8, 14)]:
                    if test_count >= max_tests:
                        break
                    
                    test_multipliers = {
                        (1, 2): mult_1_2,
                        (3, 4): mult_3_4,
                        (5, 7): mult_5_7,
                        (8, 14): mult_8_14,
                        (15, 30): base_multipliers[(15, 30)]
                    }
                    
                    config = {'range_multipliers': test_multipliers}
                    calculator = PerfectScoreCalculator(config)
                    
                    total_error = 0
                    exact_matches = 0
                    
                    for case in data:
                        days = case['input']['trip_duration_days']
                        miles = case['input']['miles_traveled']
                        receipts = case['input']['total_receipts_amount']
                        expected = case['expected_output']
                        
                        predicted = calculator.calculate(days, miles, receipts)
                        error = abs(predicted - expected)
                        total_error += error
                        
                        if error <= 0.01:
                            exact_matches += 1
                    
                    avg_error = total_error / len(data)
                    score = avg_error * 100 + (1000 - exact_matches) * 0.1
                    
                    if score < best_score:
                        best_score = score
                        best_config = test_multipliers.copy()
                        print(f"New best: Score {score:.0f}, Avg error ${avg_error:.2f}, {exact_matches} exact")
                        print(f"  Multipliers: {mult_1_2:.2f}, {mult_3_4:.2f}, {mult_5_7:.2f}, {mult_8_14:.2f}")
                    
                    test_count += 1
                    if test_count % 50 == 0:
                        print(f"Tested {test_count} combinations...")
    
    print(f"\nFinal best score: {best_score:.0f}")
    print(f"Best multipliers: {best_config}")
    return best_config, best_score

def test_combined_strategies():
    """Test combinations of optimizations"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n\nCOMBINED STRATEGY TESTING")
    print("=" * 60)
    
    # Get best range multipliers from fine-tuning
    best_multipliers, _ = fine_tune_range_multipliers()
    
    strategies = [
        {"name": "Current Best", "config": {"range_multipliers": best_multipliers}},
        {"name": "Best + Nickel", "config": {"range_multipliers": best_multipliers, "use_nickel_rounding": True}},
        {"name": "Best + Global 1.01", "config": {"range_multipliers": best_multipliers, "global_adjustment": 1.01}},
        {"name": "Best + Global 0.99", "config": {"range_multipliers": best_multipliers, "global_adjustment": 0.99}},
        {"name": "Triple Combo", "config": {"range_multipliers": best_multipliers, "use_nickel_rounding": True, "global_adjustment": 1.005}},
    ]
    
    results = []
    
    for strategy in strategies:
        calculator = PerfectScoreCalculator(strategy["config"])
        
        total_error = 0
        exact_matches = 0
        close_matches = 0
        
        for case in data:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            predicted = calculator.calculate(days, miles, receipts)
            error = abs(predicted - expected)
            total_error += error
            
            if error <= 0.01:
                exact_matches += 1
            if error <= 1.00:
                close_matches += 1
        
        avg_error = total_error / len(data)
        score = avg_error * 100 + (1000 - exact_matches) * 0.1
        
        results.append({
            'name': strategy['name'],
            'score': score,
            'avg_error': avg_error,
            'exact_matches': exact_matches,
            'close_matches': close_matches
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'])
    
    print("Strategy Performance:")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:15}: Score {r['score']:7.0f}, ${r['avg_error']:6.2f} avg, {r['exact_matches']:2d} exact, {r['close_matches']:3d} close")
    
    return results[0]  # Return best strategy

def analyze_remaining_worst_cases(best_config):
    """Analyze worst cases under the optimized model"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n\nANALYZING REMAINING WORST CASES")
    print("=" * 60)
    
    calculator = PerfectScoreCalculator(best_config['config'] if 'config' in best_config else best_config)
    
    errors = []
    
    for i, case in enumerate(data):
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        
        predicted = calculator.calculate(days, miles, receipts)
        error = abs(predicted - expected)
        
        errors.append({
            'case': i,
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'expected': expected,
            'predicted': predicted,
            'error': error,
            'ratio': expected / predicted if predicted > 0 else 0
        })
    
    # Sort by error
    errors.sort(key=lambda x: x['error'], reverse=True)
    
    print("Top 15 remaining worst cases:")
    print("-" * 60)
    for i, err in enumerate(errors[:15]):
        print(f"{i+1:2d}. Case {err['case']:3d}: {err['days']:2d}d, {err['miles']:4.0f}mi, ${err['receipts']:7.2f}")
        print(f"     Expected: ${err['expected']:7.2f}, Got: ${err['predicted']:7.2f}, Error: ${err['error']:6.2f}")
    
    # Look for patterns in remaining errors
    print(f"\nPattern analysis of worst 50 cases:")
    worst_50 = errors[:50]
    
    day_distribution = defaultdict(int)
    ratio_distribution = defaultdict(int)
    
    for err in worst_50:
        day_distribution[err['days']] += 1
        ratio_bucket = round(err['ratio'], 1)
        ratio_distribution[ratio_bucket] += 1
    
    print("Day distribution:", dict(day_distribution))
    print("Ratio distribution:", dict(ratio_distribution))

def main():
    """Run comprehensive perfect score optimization"""
    
    print("🎯 PERFECT SCORE OPTIMIZATION SUITE")
    print("=" * 60)
    
    # Step 1: Fine-tune range multipliers
    best_multipliers, fine_tuned_score = fine_tune_range_multipliers()
    
    # Step 2: Test combined strategies
    best_strategy = test_combined_strategies()
    
    # Step 3: Analyze remaining issues
    analyze_remaining_worst_cases(best_strategy)
    
    print(f"\n🏆 OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best strategy: {best_strategy['name']}")
    print(f"Final score: {best_strategy['score']:.0f}")
    print(f"Exact matches: {best_strategy['exact_matches']}")
    print(f"Average error: ${best_strategy['avg_error']:.2f}")
    
    original_score = 16840
    improvement = original_score - best_strategy['score']
    print(f"Total improvement: {improvement:.0f} points ({improvement/original_score*100:.1f}%)")

if __name__ == "__main__":
    main() 