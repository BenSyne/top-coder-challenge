# 🔍 Black Box Legacy Reimbursement System - Comprehensive Project Plan

## 📋 Table of Contents
1. [Problem Analysis](#problem-analysis)
2. [Data Intelligence Gathering](#data-intelligence-gathering)
3. [Mathematical Model Hypothesis](#mathematical-model-hypothesis)
4. [Implementation Strategy](#implementation-strategy)
5. [Testing & Validation Strategy](#testing-validation-strategy)
6. [Optimization Plan](#optimization-plan)
7. [Risk Analysis & Mitigation](#risk-analysis-mitigation)

---

## 🎯 Problem Analysis

### Core Challenge
We need to reverse-engineer a 60-year-old travel reimbursement calculation system that:
- Has NO documentation or source code access
- Contains intentional bugs/quirks that must be preserved
- Has been modified over decades with unknown logic layers
- Is still actively used by ACME Corp daily

### Available Resources
1. **1,000 historical input/output examples** (public_cases.json)
   - Input: trip_duration_days (int), miles_traveled (int), total_receipts_amount (float)
   - Output: reimbursement_amount (float, 2 decimal places)

2. **5 employee interviews** revealing:
   - Anecdotal patterns and theories
   - Contradictory observations
   - Suspected bugs and quirks
   - Different departmental experiences

3. **5,000 blind test cases** (private_cases.json) for final validation

### Success Criteria
- **Perfect**: ±$0.01 accuracy on all 1,000 public cases
- **Excellent**: 950+ exact matches
- **Great**: 800+ exact matches
- Must handle 5,000 private cases without knowing expected outputs

---

## 🔬 Data Intelligence Gathering

### From Employee Interviews - Consolidated Insights

#### 1. **Base Per Diem Patterns**
- **Lisa (Accounting)**: $100/day base rate confirmed
- **Marcus (Sales)**: Variations exist but base seems consistent
- **5-day bonus**: Multiple employees confirm special treatment for exactly 5-day trips

#### 2. **Mileage Calculation Tiers**
- **First 100 miles**: ~$0.58/mile (standard federal rate)
- **Beyond 100 miles**: Diminishing returns, possibly logarithmic
- **Efficiency bonus**: 180-220 miles/day is optimal (Kevin's data)
- **High mileage penalty**: >400 miles/day seen as unrealistic

#### 3. **Receipt Processing Complexity**
- **Small receipt penalty**: <$50 can result in LESS than base per diem
- **Sweet spot**: $600-800 total receipts
- **Daily spending targets**:
  - Short trips (1-3 days): $75/day
  - Medium trips (4-6 days): $120/day
  - Long trips (7+ days): $90/day
- **Rounding bug**: Receipts ending in .49 or .99 get bonus

#### 4. **Trip Category Behaviors**
- **Kevin's 6 calculation paths**:
  1. Quick high-mileage trips
  2. Long low-mileage trips
  3. Medium balanced trips
  4. Efficiency-optimized trips
  5. High-spending trips
  6. Minimal-spending trips

#### 5. **Interaction Effects**
- Trip length × miles/day efficiency
- Spending/day × total mileage
- Receipt total × trip duration
- Possible seasonal/timing effects (disputed)

#### 6. **Known Bugs/Quirks**
- Rounding bug on .49/.99 cents
- Small receipt penalty
- 5-day trip bonus inconsistency
- Possible user history tracking (Marcus's theory)
- Potential day-of-week effects (Kevin's data)

---

## 🧮 Mathematical Model Hypothesis

### Primary Calculation Components

```python
reimbursement = (
    base_per_diem(days) +
    mileage_reimbursement(miles, days) +
    receipt_reimbursement(receipts, days) +
    efficiency_bonus(miles, days) +
    interaction_adjustments() +
    bug_implementations()
)
```

### Detailed Component Breakdown

#### 1. **Base Per Diem**
```python
base_per_diem = days * 100
if days == 5:
    base_per_diem += FIVE_DAY_BONUS  # Need to determine value
```

#### 2. **Mileage Reimbursement**
```python
if miles <= 100:
    mileage_reimb = miles * 0.58
else:
    # Tiered or logarithmic decay
    first_100 = 100 * 0.58
    remaining = diminishing_returns_function(miles - 100)
    mileage_reimb = first_100 + remaining
```

#### 3. **Receipt Reimbursement**
```python
daily_spending = receipts / days

# Check for penalties and bonuses
if receipts < 50 and days > 1:
    receipt_reimb = PENALTY_AMOUNT
elif 600 <= receipts <= 800:
    receipt_reimb = receipts * OPTIMAL_RATE
else:
    receipt_reimb = complex_receipt_function(receipts, days)

# Apply rounding bug
if str(receipts).endswith('.49') or str(receipts).endswith('.99'):
    receipt_reimb += ROUNDING_BONUS
```

#### 4. **Efficiency Calculations**
```python
miles_per_day = miles / days

if 180 <= miles_per_day <= 220:
    efficiency_bonus = OPTIMAL_EFFICIENCY_BONUS
elif miles_per_day > 400:
    efficiency_bonus = PENALTY_FOR_UNREALISTIC_TRAVEL
else:
    efficiency_bonus = efficiency_curve(miles_per_day)
```

#### 5. **Category Detection**
```python
def determine_trip_category(days, miles, receipts):
    miles_per_day = miles / days
    spending_per_day = receipts / days
    
    # Logic to classify into one of 6 categories
    # Each category has different calculation rules
```

---

## 💻 Implementation Strategy

### Phase 1: Data Analysis & Pattern Discovery (Days 1-2)

1. **Load and Explore Data**
   ```python
   # analyze_data.py
   - Load all 1,000 public cases
   - Basic statistics: min, max, mean, std for each input/output
   - Correlation analysis between inputs and outputs
   - Distribution plots for each variable
   ```

2. **Feature Engineering**
   ```python
   # Create derived features
   - miles_per_day = miles / days
   - spending_per_day = receipts / days
   - efficiency_ratio = miles / receipts
   - trip_category = categorize_trip()
   ```

3. **Pattern Detection**
   ```python
   # Find specific patterns mentioned in interviews
   - Identify all 5-day trips and their bonus pattern
   - Plot mileage vs reimbursement to find tiers
   - Analyze receipt ending patterns (.49, .99)
   - Look for threshold effects
   ```

### Phase 2: Model Building (Days 3-4)

1. **Component Implementation**
   ```python
   # calculate_reimbursement.py
   class ReimbursementCalculator:
       def calculate_base_per_diem(self, days)
       def calculate_mileage(self, miles, days)
       def calculate_receipts(self, receipts, days)
       def calculate_efficiency_bonus(self, miles, days)
       def apply_category_rules(self, category, base_calc)
       def apply_bugs(self, amount, receipts)
   ```

2. **Parameter Tuning**
   - Use optimization algorithms to find optimal parameters
   - Grid search for threshold values
   - Regression analysis for continuous functions

3. **Rule Discovery**
   - Decision tree to find categorical rules
   - Regression to find continuous relationships
   - Ensemble methods to capture complex interactions

### Phase 3: Testing & Refinement (Days 5-6)

1. **Iterative Testing**
   ```bash
   ./eval.sh  # Run after each change
   # Focus on:
   - Exact matches (±$0.01)
   - High-error cases
   - Pattern in errors
   ```

2. **Error Analysis**
   - Group errors by trip characteristics
   - Identify systematic biases
   - Refine model based on error patterns

3. **Edge Case Handling**
   - Very short/long trips
   - Extreme mileage values
   - Unusual receipt amounts
   - Boundary conditions

### Phase 4: Final Optimization (Day 7)

1. **Performance Optimization**
   - Profile code for bottlenecks
   - Optimize calculations
   - Consider caching if needed

2. **Final Validation**
   - Run full test suite
   - Generate private results
   - Document findings

---

## 🧪 Testing & Validation Strategy

### 1. **Unit Testing**
```python
# test_components.py
- Test each calculation component independently
- Verify known patterns (5-day bonus, rounding bug)
- Test edge cases
```

### 2. **Integration Testing**
```python
# test_integration.py
- Test full calculation pipeline
- Verify category detection
- Test interaction effects
```

### 3. **Regression Testing**
- Save successful test results
- Ensure changes don't break working cases
- Track improvement over iterations

### 4. **Statistical Validation**
- Compare distributions of calculated vs expected
- Analyze error patterns
- Look for systematic biases

---

## ⚡ Optimization Plan

### If Performance Issues Arise:

1. **Python Optimizations**
   - Use NumPy for vectorized calculations
   - Implement caching for repeated calculations
   - Profile and optimize hot paths

2. **Algorithm Optimizations**
   - Pre-compute category classifications
   - Use lookup tables for common values
   - Minimize function calls

3. **Language Migration** (if necessary)
   - Port to Go/Rust for 10-100x speedup
   - Keep Python for prototyping
   - Use same algorithm in faster language

---

## ⚠️ Risk Analysis & Mitigation

### 1. **Overfitting Risk**
- **Risk**: Model perfectly fits public cases but fails on private
- **Mitigation**: Use cross-validation, look for general patterns

### 2. **Missing Pattern Risk**
- **Risk**: Employees missed mentioning crucial patterns
- **Mitigation**: Systematic data analysis beyond interview hints

### 3. **Complex Interaction Risk**
- **Risk**: Interactions too complex to model
- **Mitigation**: Use machine learning alongside rule-based approach

### 4. **Performance Risk**
- **Risk**: Complex model too slow for 5,000 cases
- **Mitigation**: Profile early, optimize continuously, have backup plan

### 5. **Edge Case Risk**
- **Risk**: Private cases contain unseen edge cases
- **Mitigation**: Robust error handling, reasonable defaults

---

## 📊 Key Metrics to Track

1. **Accuracy Metrics**
   - Exact matches (±$0.01)
   - Close matches (±$1.00)
   - Mean absolute error
   - Max error cases

2. **Performance Metrics**
   - Time per calculation
   - Memory usage
   - Bottleneck identification

3. **Pattern Metrics**
   - Coverage of each trip category
   - Accuracy by category
   - Error distribution

---

## 🎯 Implementation Priorities

### Must-Have Features (Week 1)
1. Base per diem calculation
2. Tiered mileage reimbursement
3. Receipt processing with penalties
4. 5-day bonus
5. Rounding bug (.49/.99)
6. Basic efficiency calculations

### Should-Have Features (If Time Permits)
1. All 6 category classifications
2. Complex interaction effects
3. All mentioned edge cases
4. Time-based variations

### Nice-to-Have Features
1. ML-based pattern detection
2. Automated parameter optimization
3. Confidence scoring for predictions

---

## 📝 Next Steps

1. **Immediate Actions**
   - Set up Python project structure
   - Create data analysis notebook
   - Begin pattern exploration

2. **Day 1 Goals**
   - Complete initial data analysis
   - Identify obvious patterns
   - Build basic calculator skeleton

3. **Success Checkpoint**
   - By end of Day 2: 500+ exact matches
   - By end of Day 4: 800+ exact matches
   - By end of Day 6: 950+ exact matches

---

## 🔧 Tools & Technologies

### Required
- Python 3.x
- NumPy & Pandas for data analysis
- Matplotlib/Seaborn for visualization
- SciPy for optimization

### Optional
- Scikit-learn for ML approaches
- Jupyter notebooks for exploration
- Git for version control
- Performance profilers

---

## 📚 Reference Materials

### Key Interview Insights to Remember
1. Kevin's spreadsheets show real patterns
2. Lisa sees the numbers daily - trust her observations
3. Marcus's experience with variations is real
4. The system evolved over 60 years - expect layers
5. Bugs are features now - preserve them

### Critical Numbers
- Base per diem: $100/day
- Standard mileage: $0.58/mile (first 100)
- Optimal efficiency: 180-220 miles/day
- Receipt sweet spot: $600-800
- 5-day trips: Special bonus (amount TBD)

---

## 📊 Phase 1 Analysis Results (COMPLETED)

### Key Discoveries from Data Analysis

#### 1. **Base Per Diem - SURPRISE FINDING**
- Confirmed ~$100/day base rate
- BUT: 5-day trips show LOWER per-day rate ($254.52/day) not higher!
- This contradicts interview expectations - possible negative bonus?
- Per-day rates decrease consistently with trip length

#### 2. **Mileage Tiers - CONFIRMED**
- Clear tiered structure with diminishing returns:
  - 0-50 miles: ~$12.83/mile (extremely high!)
  - 50-100 miles: ~$3.38/mile
  - 100-200 miles: ~$2.84/mile
  - 200-300 miles: ~$1.89/mile
  - 300-500 miles: ~$1.36/mile
  - 500-1000 miles: ~$0.99/mile
- NOT the expected $0.58/mile base rate!

#### 3. **Rounding Bug - OPPOSITE EFFECT**
- Cases ending in .49/.99: Average $566.19
- Normal cases: Average $1,373.33
- The "bug" actually REDUCES reimbursement by ~$800!
- Need to investigate why this is opposite of interviews

#### 4. **Efficiency Sweet Spot - CONFIRMED**
- 180-220 miles/day shows $64.93/day bonus
- Clear efficiency rewards for optimal travel pace
- Very high miles/day (400+) gets penalized

#### 5. **Receipt Patterns**
- Small receipts (<$50): $149.50/day average
- Sweet spot ($600-800): $1,141.87 average
- Vacation penalty confirmed: 8+ days with high spending gets lower daily rate

#### 6. **Correlation Analysis**
- Strongest predictor: receipts (0.704)
- Days traveled (0.514)
- Miles traveled (0.432)
- Spending per day has low correlation (0.051)
- Miles per day has NEGATIVE correlation (-0.142)

### Critical Contradictions to Resolve
1. Why do 5-day trips get LESS not MORE per day?
2. Why does rounding bug REDUCE not INCREASE reimbursement?
3. Why is base mileage rate so different from federal $0.58?

### Next Steps for Phase 2
- Build initial model based on these findings
- Investigate the contradictions
- Look for hidden factors causing unexpected patterns
- Consider that interviews may have been misleading or incomplete

---

*This plan is a living document. Update as new patterns are discovered.*

## Progress Summary
- **Best Score Achieved**: 17,295 (average error: $171.95)
- **Model**: Linear regression with caps and adjustments
- **Key Discovery**: System likely uses a simple linear formula with special handling for edge cases

## Latest Findings

### Successful Formula Components
1. **Base Linear Model** (from regression analysis):
   - Intercept: $266.71
   - Days coefficient: $50.05
   - Miles coefficient: $0.4456
   - Receipts coefficient: $0.3829
   - R-squared: 0.7840

2. **Critical Adjustments**:
   - **Rounding bug factor**: 0.457 (for receipts ending in .49/.99)
   - **Receipt caps**: Above $1800, receipts processed at reduced rate
   - **Mileage caps**: Above 800 miles, reduced rate applies
   - **5-day trip penalty**: ~8% reduction
   - **Efficiency bonus**: $30 for 180-220 miles/day

3. **Systematic Patterns**:
   - We're over-predicting 92.7% of cases
   - Mid-length trips (7-9 days) with high mileage are problematic
   - Receipt processing appears tiered but implementing tiers made results worse
   - Simple formulas (e.g., `150 + 50*d + 0.45*m + 0.40*r`) perform surprisingly well

### Problem Areas
1. **High mileage + high receipt cases** (consistently under-predicted by $600-800):
   - Example: 7 days, 1006 miles, $1181 receipts → Expected $2280, we predict ~$1450

2. **No exact matches achieved** - suggests we're missing a discrete rule or rounding method

## Phase 7: Final Optimization

### Strategy
1. **Hybrid approach**: Use simple base formula with targeted adjustments
2. **Focus on edge cases**: Special handling for high-value trips
3. **Test rounding strategies**: The system might round intermediate calculations
4. **Consider discrete buckets**: Reimbursement might jump at specific thresholds

### Next Steps
1. Implement simplified base formula with refined edge case handling
2. Test different rounding approaches (ceil, floor, round to nearest $5/$10)
3. Add special boost for mid-length high-value trips
4. Consider that some trip categories might use entirely different formulas

### Key Insights from Interviews
- **Mileage sweet spot** exists (180-220 miles/day) ✓ Implemented
- **Rounding bug** reduces reimbursement ✓ Confirmed (factor 0.457)
- **5-day trips** special handling ✓ Confirmed (but penalty, not bonus)
- **Receipt thresholds** exist ✓ Confirmed but complex to implement
- **Very generous for short trips** - Partially implemented

### Technical Notes
- All 1000 test cases run successfully
- Execution time is fast (<1 second for all cases)
- No memory or performance issues
- Ready for final submission once exact matches are achieved 