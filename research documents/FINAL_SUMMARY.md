# Black Box Legacy Reimbursement System - Final Solution

## Challenge Overview
Reverse-engineer a 60-year-old travel reimbursement system using 1,000 historical examples to create a perfect replica.

## Final Solution: ML Ensemble Model

### Score: 8,581.81
- **Average Error**: $85.82 per case
- **Improvement**: 49% better than baseline (16,840)
- **Model**: Machine Learning Ensemble with Surgical Baseline

## Key Components

### 1. Surgical Baseline Model
Our proven formula with strategic adjustments:
- Base formula: $266.71 + $50.05×days + $0.4456×miles + $0.3829×receipts
- Rounding bug detection (receipts ending in .49/.99): ×0.457
- Receipt/mile caps to prevent extreme values
- Range-specific multipliers by trip length
- Surgical fixes for edge cases

### 2. ML Ensemble Enhancement
Three models trained to predict residuals:
- **Random Forest** (100 estimators)
- **Gradient Boosting** (100 estimators) 
- **Neural Network** (100,50 hidden layers)

### 3. Feature Engineering
35+ engineered features including:
- Polynomial features (days², miles×receipts, etc.)
- Categorical encodings (trip length bins, mileage bins)
- Special pattern flags (rounding bug, efficiency bonus)
- Log transformations for scale normalization

## Key Discoveries

1. **Rounding Bug**: Receipts ending in .49/.99 trigger 54% reduction
2. **Tiered Mileage**: Not standard rate - uses 4-tier system
3. **Efficiency Bonus**: 180-220 miles/day adds ~$30
4. **Trip Length Bias**: Systematic under-prediction for 5-7 day trips
5. **Legacy Patterns**: Special multipliers for edge cases

## Files Submitted
- `calculate_reimbursement.py` - ML ensemble implementation
- `private_results.txt` - 5,000 predictions for private test set
- `private_answers.txt` - Detailed iteration history

## Technical Notes
- Initial training takes ~10-15 minutes
- Falls back to surgical model if ML libraries unavailable
- Requires sklearn, numpy for full functionality
- Average prediction time: <100ms per case after training

## Results Evolution
1. **Iteration 1**: Linear model → Score 16,840
2. **Iteration 2**: Surgical optimization → Score 12,710 
3. **Iteration 3**: ML ensemble → Score 8,581.81

---
*Solution achieves 49% improvement through combining domain expertise (surgical model) with machine learning (ensemble residual prediction)* 