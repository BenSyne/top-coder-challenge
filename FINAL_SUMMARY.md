# Black Box Reimbursement System - Final Solution Summary

## Final Score: 16,840 (Average Error: $167.40)

### Solution Overview
After extensive analysis and testing, the reimbursement system appears to use a **linear formula with special adjustments** for edge cases.

### Core Formula
```
Base = 266.71 + 50.05×days + 0.4456×miles + 0.3829×receipts
```

### Key Adjustments

1. **Caps on High Values**:
   - Miles > 800: Reduced rate (0.25× for excess)
   - Receipts > $1800: Reduced rate (0.15× for excess)

2. **Rounding Bug**: 
   - Receipts ending in .49 or .99: Total × 0.457

3. **Trip-Specific Multipliers**:
   - 5-day trips: × 0.92
   - 7-8 day trips (high value): × 1.25
   - 12+ day trips (high mileage): × 0.85
   - 13-14 day trips (high value): × 1.20
   - Short trips (<2 days, minimal inputs): × 1.15

4. **Efficiency Bonus**: 
   - 180-220 miles/day: +$30

### Key Findings

1. **Employee Interviews vs Reality**:
   - ✓ Rounding bug exists (but reduces, not increases)
   - ✓ Efficiency sweet spot confirmed
   - ✗ 5-day trips get penalty, not bonus
   - ✓ Receipt thresholds exist
   - ✓ Mileage isn't standard rate

2. **System Characteristics**:
   - Not a simple lookup table
   - Uses continuous calculations with discrete adjustments
   - High-value trips follow different rules
   - No exact matches achieved (suggests additional complexity)

3. **Remaining Challenges**:
   - 12-13 day trips with 1000+ miles still under-predicted
   - No exact matches (system may use different rounding)
   - Very high receipt cases ($2000+) are complex

### Technical Implementation
- Language: Python 3
- Execution time: <1ms per calculation
- All 1000 test cases run successfully
- No external dependencies required

### Lessons Learned
1. Simple models often outperform complex ones
2. Edge cases require specific handling
3. Employee interviews provide clues but not complete picture
4. Systematic analysis of errors reveals patterns
5. Legacy systems often have "bugs" that become features

### Future Improvements
1. Analyze exact match patterns more deeply
2. Test alternative rounding strategies
3. Consider that different trip categories might use entirely different formulas
4. Investigate if there's a maximum reimbursement cap

---

This solution represents ~8 hours of analysis, testing, and refinement across 7 major iterations. 