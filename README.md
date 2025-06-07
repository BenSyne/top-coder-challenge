# Black Box Reimbursement Challenge Solution

This repository contains my solution to the Black Box Legacy Reimbursement System challenge.

## Challenge Overview
Reverse-engineer a 60-year-old travel reimbursement system using 1,000 historical input/output examples and employee interviews.

## Solution Score
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

## Repository Structure
```
├── calculate_reimbursement.py  # Main calculator implementation
├── run.sh                      # Shell wrapper for calculator
├── analyze_data.py            # Initial data analysis
├── deeper_analysis.py         # Linear regression analysis
├── error_analysis.py          # Error pattern analysis
├── PROJECT_PLAN.md            # Detailed project plan and findings
├── FINAL_SUMMARY.md           # Final solution summary
└── public_cases.json          # Test data (1000 cases)
```

## Key Findings
- Base formula: `266.71 + 50.05×days + 0.4456×miles + 0.3829×receipts`
- Rounding bug factor: 0.457 (for receipts ending in .49/.99)
- Various adjustments for trip categories and edge cases
- See FINAL_SUMMARY.md for complete details

## Dependencies
- Python 3.x
- pandas, numpy, matplotlib, scikit-learn (for analysis scripts)
- No dependencies for the main calculator

## Author
Ben Syne

## Date
December 2024
