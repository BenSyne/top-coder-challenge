#!/usr/bin/env python3
"""
Surgical Optimizer - Target specific patterns in worst cases
"""

import json
import sys

class SurgicalOptimizedCalculator:
    """Calculator with surgical fixes for remaining worst cases"""
    
    def __init__(self):
        # Optimized coefficients and multipliers
        self.intercept = 266.71
        self.coef_days = 50.05
        self.coef_miles = 0.4456
        self.coef_receipts = 0.3829
        self.rounding_bug_factor = 0.457
        
        # Fine-tuned range multipliers
        self.range_multipliers = {
            (1, 2): 1.06,
            (3, 4): 1.09,
            (5, 7): 1.14,
            (8, 14): 1.01,
            (15, 30): 1.05
        }
        
    def get_range_multiplier(self, days):
        """Get range multiplier for trip length"""
        for (min_days, max_days), multiplier in self.range_multipliers.items():
            if min_days <= days <= max_days:
                return multiplier
        return 1.00
        
    def calculate(self, days, miles, receipts):
        """Calculate with surgical optimizations"""
        
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
        
        # SURGICAL FIXES based on worst case analysis
        if not has_rounding_bug:
            
            # Fix 1: High-receipt 7-day trips (over-predicted cases)
            if days == 7 and receipts > 2000:
                amount *= 0.85  # Reduce by 15%
            
            # Fix 2: 8-14 day trips with moderate receipts (under-predicted)
            if 8 <= days <= 14 and 900 <= receipts <= 1500 and miles < 1200:
                amount *= 1.15  # Boost by 15%
            
            # Fix 3: Very long trips with high mileage (pattern from worst cases)
            if days >= 12 and miles >= 1000 and receipts < 1200:
                amount *= 1.10  # Additional boost beyond existing 0.85 factor
            
            # Fix 4: Medium-length high-mileage trips
            if 8 <= days <= 11 and miles >= 1000 and receipts >= 1000:
                amount *= 0.95  # Slight reduction
        
        result = round(amount, 2)
        return max(0.0, result)

def test_surgical_optimizations():
    """Test surgical optimizations on worst cases"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("🔧 TESTING SURGICAL OPTIMIZATIONS")
    print("=" * 60)
    
    calculator = SurgicalOptimizedCalculator()
    
    total_error = 0
    exact_matches = 0
    close_matches = 0
    
    # Track before/after for worst cases
    worst_case_indices = [477, 528, 908, 334, 642, 694, 297, 985, 971, 813, 133, 132, 48, 870, 144]
    worst_case_improvements = []
    
    for i, case in enumerate(data):
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
        
        # Track worst case improvements
        if i in worst_case_indices:
            worst_case_improvements.append({
                'case': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'expected': expected,
                'predicted': predicted,
                'error': error
            })
    
    avg_error = total_error / len(data)
    score = avg_error * 100 + (1000 - exact_matches) * 0.1
    
    print(f"Surgical optimization results:")
    print(f"Score: {score:.0f}")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Exact matches: {exact_matches}")
    print(f"Close matches: {close_matches}")
    
    prev_score = 14377
    improvement = prev_score - score
    print(f"Improvement: {improvement:.0f} points")
    
    # Show improvements on former worst cases
    print(f"\nImprovements on former worst cases:")
    print("-" * 60)
    for imp in sorted(worst_case_improvements, key=lambda x: x['error'])[:10]:
        print(f"Case {imp['case']:3d}: {imp['days']:2d}d, {imp['miles']:4.0f}mi, ${imp['receipts']:7.2f}")
        print(f"         Expected: ${imp['expected']:7.2f}, Got: ${imp['predicted']:7.2f}, Error: ${imp['error']:6.2f}")
    
    return score

def test_adaptive_corrections():
    """Test more adaptive correction strategies"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n\nTESTING ADAPTIVE CORRECTIONS")
    print("=" * 60)
    
    # Test different correction strategies
    strategies = [
        ("Base surgical", lambda d, m, r, base: base),
        ("Receipt ratio", lambda d, m, r, base: base * (1 - 0.05 * max(0, (r - 2000) / 1000)) if d == 7 else base),
        ("Miles efficiency", lambda d, m, r, base: base * (1 + 0.02 * max(0, min(2, (200 - m/d) / 50))) if d >= 8 else base),
        ("Dynamic range", lambda d, m, r, base: base * (1.05 if r/d < 100 else 0.95) if 8 <= d <= 14 else base),
    ]
    
    best_score = float('inf')
    best_strategy = None
    
    for strategy_name, correction_func in strategies:
        total_error = 0
        exact_matches = 0
        
        for case in data:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            
            # Use base calculator then apply correction
            calculator = SurgicalOptimizedCalculator()
            
            # Get base amount before final processing
            receipt_str = f"{receipts:.2f}"
            has_rounding_bug = receipt_str.endswith('.49') or receipt_str.endswith('.99')
            
            capped_receipts = receipts
            if receipts > 1800:
                capped_receipts = 1800 + (receipts - 1800) * 0.15
            
            capped_miles = miles
            if miles > 800:
                capped_miles = 800 + (miles - 800) * 0.25
            
            amount = (calculator.intercept + 
                     calculator.coef_days * days + 
                     calculator.coef_miles * capped_miles + 
                     calculator.coef_receipts * capped_receipts)
            
            if has_rounding_bug:
                amount *= calculator.rounding_bug_factor
            else:
                # Apply all existing adjustments...
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
                
                range_multiplier = calculator.get_range_multiplier(days)
                amount *= range_multiplier
                
                # Apply adaptive correction
                amount = correction_func(days, miles, receipts, amount)
            
            predicted = max(0.0, round(amount, 2))
            error = abs(predicted - expected)
            total_error += error
            
            if error <= 0.01:
                exact_matches += 1
        
        avg_error = total_error / len(data)
        score = avg_error * 100 + (1000 - exact_matches) * 0.1
        
        print(f"{strategy_name:15}: Score {score:7.0f}, ${avg_error:6.2f} avg, {exact_matches} exact")
        
        if score < best_score:
            best_score = score
            best_strategy = strategy_name
    
    print(f"\nBest adaptive strategy: {best_strategy} with score {best_score:.0f}")
    return best_score

def main():
    """Test surgical optimizations"""
    
    print("🎯 SURGICAL OPTIMIZATION FOR PERFECT SCORE")
    print("=" * 60)
    
    # Test surgical fixes
    surgical_score = test_surgical_optimizations()
    
    # Test adaptive corrections
    adaptive_score = test_adaptive_corrections()
    
    best_score = min(surgical_score, adaptive_score)
    
    print(f"\n🏆 SURGICAL OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best surgical score: {surgical_score:.0f}")
    print(f"Best adaptive score: {adaptive_score:.0f}")
    print(f"Best overall score: {best_score:.0f}")
    
    original_score = 16840
    total_improvement = original_score - best_score
    print(f"Total improvement: {total_improvement:.0f} points ({total_improvement/original_score*100:.1f}%)")

if __name__ == "__main__":
    main() 