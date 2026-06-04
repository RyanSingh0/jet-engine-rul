# Jet Engine Predictive Maintenance (NASA C-MAPSS)

> **Statistical & ML Analysis of Turbofan Engine Degradation**
> METCS 544 · Boston University

![R](https://img.shields.io/badge/R-4.3-blue)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-lightgrey)
![AUC](https://img.shields.io/badge/Failure%20AUC-0.945-brightgreen)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.52%25-green)

---

## Overview

Statistical and ML analysis of NASA's C-MAPSS turbofan engine run-to-failure dataset, predicting **Remaining Useful Life (RUL)** of aircraft engines and classifying **imminent failure** from sensor time-series.

**Research questions:**
1. Which sensor measurements change significantly as an engine approaches failure?
2. Can we predict remaining useful life or imminent failure from sensor data?

---

## Dataset

| Property | Value |
|----------|-------|
| Source | NASA C-MAPSS FD001 (Prognostics Data Repository) |
| Engines | 100 training + 100 test engines |
| Sensors | 21 per engine per cycle |
| Scenario | Single operating condition (sea-level), high-pressure compressor degradation |
| Analysis sample | 1,000 observations (sampled from ~20,000) |
| Excluded sensors | s1, s10, s18, s19, setting3 (zero variance in FD001) |

**Key variables:**
- `Cycle`: Operating time index per engine
- `Sensor 11 (s11)`: Increases as failure approaches (r = −0.70 with RUL)
- `Sensor 12 (s12)`: Decreases with degradation (r = +0.67 with RUL, inverse)
- `RUL`: Remaining useful life in cycles
- `FailSoon`: Binary — RUL ≤ 30 cycles (imminent failure indicator)

---

## Statistical Analysis

### Sensor Degradation — Key Findings

Sensor 11 is the primary degradation indicator:
- Early-stage mean: **47.44**
- Near-failure mean: **47.87**
- Difference: +0.43 units (+0.9%)

This small absolute change is highly statistically significant and physically meaningful.

### Hypothesis Tests

| Test | Result | Finding |
|------|--------|---------|
| One-sample t-test (s11 vs baseline) | t = 23.59, p < 0.001 | s11 significantly elevated vs healthy baseline |
| Two-sample t-test (early vs late stage) | t = 26.10, p < 0.001 | Near-failure engines have significantly higher s11 |
| One-way ANOVA (4 life-stage quartiles) | F(3,996) = 343.7, p < 0.001 | All quartile pairs differ significantly (Tukey HSD) |
| Two-way ANOVA (Engine ID × Stage) | Stage F ≈ 3165, p < 0.001 | Consistent degradation direction; magnitude varies by engine |
| ANCOVA (LifeCategory + Cycle) | Interaction p < 2e-12 | Short-life engines degrade **faster per cycle** than long-life engines |

---

## Regression Models

### RUL Prediction (Regression)

| Model | Predictors | R² | Notes |
|-------|-----------|-----|-------|
| Simple LR | Sensor 11 | 0.423 | Strong single predictor |
| Simple LR | Cycle only | 0.466 | Intuitive baseline |
| **Multiple LR** | **s11 + s9** | **0.452** | Best linear model |

Sensor 9 (s9) chosen as second predictor: moderate correlation with RUL (r ≈ −0.39), low collinearity with s11 (r ~ 0.27).

Residual diagnostics showed mild heteroscedasticity — model struggles at extremes (very low and very high RUL), suggesting non-linear degradation patterns near failure.

### Failure Classification (Logistic Regression)

**Defining FailSoon:** RUL ≤ 30 cycles = imminent failure (22.5% of sample)

| Model | Predictors | AUC | Test Accuracy |
|-------|-----------|-----|--------------|
| Simple logistic | s11 only | — | — |
| **Regularized (Lasso)** | **s11 + multi-sensor** | **0.945** | **96.52%** |
| Ridge | multi-sensor | 0.942 | — |
| Elastic Net | multi-sensor | 0.945 | — |

Lasso selected for final model: equal AUC to Elastic Net, with added benefit of automatic feature selection (several sensors zeroed out, confirming s11 and s12 dominance).

**Confusion matrix (regularized logistic, test set):**
| | Predicted No Fail | Predicted Fail |
|--|------------------|---------------|
| **Actual No Fail** | 12,403 TN | 95 FP |
| **Actual Fail** | 361 FN | 237 TP |

Specificity: 99.2% (almost never cries wolf). Sensitivity: 39.6% (catches ~40% of true imminent failures with strict threshold — tunable).

---

## Key Findings

1. **Sensor 11 is the primary health indicator** — increases monotonically and significantly as RUL decreases. Confirmed by t-tests, ANOVA, and regression weights.

2. **Short-life engines degrade faster** (ANCOVA interaction p < 2e-12) — not just different starting points, but different rates of deterioration. Maintenance schedules should be engine-specific.

3. **AUC 0.945 for binary failure prediction** with Lasso/Elastic Net — practically deployable as an early warning system.

4. **Regression R² ≈ 0.45–0.47** — substantial but not perfect. Unmodeled factors include engine-specific wear and non-linear degradation near failure.

---

## How to Run

```r
# Install dependencies
install.packages(c("ggplot2", "caret", "glmnet", "car"))

# Run analysis
source("analysis.R")
```

> **Data:** Download NASA C-MAPSS FD001 from https://data.nasa.gov and place train_FD001.txt in data/ folder. See data_note.md.

---

**Aryan Meena** · Boston University · METCS 544
