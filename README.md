# Team 69 — FPL Matchweek Point Prediction (Academic README)

**Notebook:** `MS1-FINAL-TEAM-69.ipynb`  
**Course context:** ML for Sports Analytics (Fantasy Premier League)  
**Authors:** Team 69 – FPL (Kerollos Ashraf Nagy · Yusuf Mohamed Alhegawy · Youssef Omar Ahmed Aboelela · Mohamed Alaaeldin Ibrahim Elsherif)

---

## Abstract
This notebook builds a supervised regression pipeline to **predict next‑gameweek Fantasy Premier League (FPL) points** per player using historical per‑match features. We compare a **Linear Regression (LR)** baseline against a small **Feed‑Forward Neural Network (FFNN)** and use **SHAP** (and prepared LIME blocks) for post‑hoc explainability. On an 80/20 random hold‑out split, the FFNN slightly outperforms LR with **lower MAE/RMSE** and **higher R² (~0.29)**. Per‑position error analysis shows best absolute error for **goalkeepers**, followed by **midfielders**, while **forwards** are the most challenging.

---

## Data
- **Source:** A merged FPL seasons dataset (per‑match rows), initially loaded from  
  `/kaggle/input/fantasy-football/cleaned_merged_seasons.csv` and later saved/used as `/kaggle/working/updated_dataset.csv` after engineering.
- **Granularity:** One row per player per gameweek (includes match, player, and popularity/market features).
- **Target definition:** `upcoming_total_points` = the **next gameweek** `total_points` for the same player (created via per‑player `shift(-1)` within each season and dropping last‑gameweek rows without a next week).

---

## Problem Formulation
- **Task:** Supervised **regression** — predict `upcoming_total_points` ∈ ℝ for each (player, gameweek).
- **Train/Test Split:** 80/20 random split (fixed `random_state=42`).
- **Metrics:** MAE, MSE, RMSE, R². We also report **per‑position MAE** to reflect FPL role differences.

---

## Features
Two groups are used (subset availability depends on the dataset columns):

1. **Match‑related**:  
   `minutes, goals_scored, assists, clean_sheets, goals_conceded, saves, bonus, bps, ict_index, influence, creativity, threat, yellow_cards, red_cards, own_goals, penalties_missed, penalties_saved, selected, transfers_balance, was_home, player_team_score, opp_team_score`

2. **Player‑related**:  
   `form, value, pos_DEF, pos_MID, pos_FWD, pos_GK` (one‑hot position dummies; **`pos_GK` dropped** to avoid dummy‑variable trap)

> Each feature’s inclusion is motivated by its direct or indirect contribution to FPL scoring or context (e.g., **minutes, goals/assists** for attackers; **saves/clean_sheets** for GKs/DEFs; **bps/ICT** as composite influence; **was_home** for home advantage; **team/opponent scores** for contextual strength).

---

## Data Cleaning & Preprocessing (and **why** each step)
1. **Normalize Position Labels** (e.g., `GKP → GK`)  
   *Why:* unify categorical values coming from heterogeneous season exports to ensure consistent encoding.

2. **Reconstruct Player Team & Match Context**  
   - Use **(season, fixture, was_home)** to compute `player_team_score` and `opp_team_score`.  
   *Why:* downstream points depend on team performance (clean sheets, attacking environment).

3. **Column Renames / Harmonization**  
   *Why:* multiple FPL dumps differ in naming (`season_x`, `GW`, etc.); harmonization makes feature selection reliable.

4. **Drop Non‑Predictive or Redundant Columns** (e.g., `transfers_in`, `transfers_out`)  
   *Why:* reduce noise/leakage risk and simplify the model to stable, interpretable signals.

5. **One‑Hot Encode Positions** → `pos_DEF`, `pos_MID`, `pos_FWD` (drop `pos_GK`)  
   *Why:* roles have different scoring rules and baseline expectations; drop one dummy to avoid multicollinearity.

6. **Create Target `upcoming_total_points` via Grouped `shift(-1)`**  
   *Why:* strictly predicts **next** GW points; avoids temporal leakage from future events in the same row.

7. **Scaling with `StandardScaler`** (fit on train, transform train/test)  
   *Why:* stabilize LR coefficients, speed FFNN training, and ensure features with different units don’t dominate (especially important for neural nets and distance‑based explanations).

---

## Models

### 1) Linear Regression (baseline)
- **Input:** Standardized numeric features (see above; `pos_GK` dropped).  
- **Loss/Objective:** OLS minimizing MSE; we report MAE/MSE/RMSE/R².  
- **Motivation:** Interpretable baseline; quick to train; good sanity check for feature engineering.

### 2) Feed‑Forward Neural Network (FFNN)
- **Architecture:** `Input → Dense(64, ReLU) → Dropout(0.15) → Dense(32, ReLU) → Dense(1)`  
- **Optimization:** Adam (lr=1e‑3), loss=`MSE`, metric=`MAE`  
- **Training:** `epochs=200`, `batch_size=64`, `validation_split=0.2`  
- **Callbacks:** `EarlyStopping(patience=10)`, `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e‑5)`  
- **Motivation:** Capture mild non‑linearities and interactions beyond linear additivity while remaining lightweight.

---

## Evaluation & Results

### Overall (hold‑out test set)
```
========= Overall Model Performance =========
                      MAE     MSE    RMSE      R²
Linear Regression  1.2257  4.6174  2.1488  0.2858
FFNN               1.1887  4.5902  2.1425  0.2900
```

**Observation.** The FFNN provides a **consistent but modest gain** over LR across all reported metrics; both models explain roughly **29%** of the variance (R²).

### Per‑Position MAE (lower is better)
```
Model       FFNN    Linear Regression
DEF         1.2947  1.3070
FWD         1.3477  1.4115
GK          0.8621  0.9029
MID         1.1397  1.1883
```

**Observation.** **GK** predictions are most accurate (lowest MAE) while **FWD** are hardest (highest MAE), matching FPL volatility patterns (explosive but inconsistent forwards vs. stable keeper scoring).

---

## Explainability

### SHAP (executed)
- **Global per‑position summaries** were computed on the test set.  
- **High‑impact features** (qualitative patterns):
  - **Attackers (FWD/MID):** `goals_scored`, `assists`, `minutes`, and attacking composites (`threat`, `ict_index`, `bps`) tend to dominate.
  - **Defenders:** `clean_sheets`, `minutes`, `bps/influence`, and **context** (`player_team_score`, `opp_team_score`).
  - **Goalkeepers:** `saves`, `clean_sheets`, `goals_conceded` (negative), and match context.
- **Why SHAP:** model‑agnostic, additively consistent attributions; easy to compare LR vs. FFNN behavior **per position**.

### LIME (prepared)
- LIME Tabular code blocks are included for **both LR and FFNN** (commented in the notebook for runtime stability).  
- **Why LIME:** local, human‑readable linear proxy around a chosen instance; complements SHAP’s global view.

> **Note:** The README reflects the notebook’s qualitative SHAP outputs. Precise per‑feature rankings vary by position and the chosen test subset.

---

## Inference Helper
The notebook provides a utility (`notebook_style_infer`) to **accept a raw row or dict**, align/scale features to the trained order, run the chosen model (LR/FFNN), and optionally **clamp negatives to 0** for FPL display. This enables quick “what‑if” scoring for upcoming fixtures once features are assembled.

---

## Limitations & Future Work
- **Modest R² (~0.29):** FPL outcomes are noisy; adding richer **fixture difficulty**, **rolling form windows**, and **bookmaker odds** may help.  
- **Temporal validation:** A **time‑series split** (train on early weeks, test on later weeks) would better mimic deployment than random splits.  
- **Injury/rotation data:** Probabilities of starting, fatigue, and injuries are not explicitly modeled.  
- **Position‑aware models:** Training **separate models per position** (rather than a single model with dummies) or using **GBMs/TabNet** could improve accuracy.  
- **Calibration:** Post‑hoc calibration for uncertainty estimation (prediction intervals) would be useful for decision making.

---

## Reproducibility
- **Environment:** Python, `pandas`, `numpy`, `scikit‑learn`, `tensorflow/keras`, `matplotlib`, `shap` (SHAP compatibility fixes included for `numpy` aliases).  
- **Paths:** Expect datasets under `/kaggle/input/...`; notebook persists an engineered CSV as `/kaggle/working/updated_dataset.csv`.  
- **Random seeds:** Set for NumPy and TensorFlow (`42`) where applicable.  
- **Scaling & Dummies:** Fitted **only on training** (to avoid leakage); `pos_GK` dropped to avoid dummy trap.

---

## How to Use (Quick Start)
1. Run the notebook cells in order (ensure input CSV exists in the specified path).  
2. Inspect **Data Cleaning** outputs and confirm columns (positions normalized; engineered team scores).  
3. Recreate target with per‑player `shift(-1)` and drop missing next‑week rows.  
4. Train **LR** and **FFNN**; compare metrics and plots.  
5. Use **SHAP** cells to generate per‑position summaries; optionally enable **LIME** blocks.  
6. Use the **inference helper** to score new hypothetical rows before a gameweek.

---

## Credits
Warm‑up, notebook intro, and project organization by **Team 69 – FPL**.  
Special thanks to course staff and FPL open‑data contributors.
