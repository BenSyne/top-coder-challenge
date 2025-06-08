# Black Box Reimbursement Challenge - Solution Documentation

This file documents my solution to the Black Box Legacy Reimbursement System challenge.

## Solution Overview
**🎯 Final Score: 16,840** (Average Error: $167.40)

## Quick Start
```bash
# Run evaluation
./eval.sh

# Calculate single reimbursement
python calculate_reimbursement.py <days> <miles> <receipts>

# Example
python calculate_reimbursement.py 5 300 500.00
```

## Solution Files
- `calculate_reimbursement.py` - Main calculator implementation
- `run.sh` - Shell wrapper for calculator
- `analyze_data.py` - Initial data analysis
- `deeper_analysis.py` - Linear regression analysis
- `error_analysis.py` - Error pattern analysis
- `discrete_analysis.py` - Analysis of discrete patterns
- `high_receipt_analysis.py` - High receipt case analysis
- `test_simple_formulas.py` - Simple formula testing
- `close_match_analysis.py` - Analysis of close matches
- `PROJECT_PLAN.md` - Detailed project plan and findings
- `FINAL_SUMMARY.md` - Final solution summary

## Key Formula
```
Base = 266.71 + 50.05×days + 0.4456×miles + 0.3829×receipts
```

With various adjustments for:
- Rounding bug (receipts ending in .49/.99): × 0.457
- High mileage/receipt caps
- Trip-specific multipliers
- Efficiency bonuses

## Performance
- All 1000 test cases run successfully
- Average error: $167.40
- Maximum error: $700.05
- Close matches (±$1.00): 1
- Exact matches: 0 (suggests additional complexity in the system)

## Author
Ben Syne

## Date
December 2024 