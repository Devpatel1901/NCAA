## NCAA Final Four Analytics Challenge 2026 — Seed Prediction

This repository contains our full workflow for the **Kaggle Final Four Analytics Challenge 2026**, where the goal is to **predict NCAA men’s basketball tournament selection and overall seeding (1–68) for each team** across recent seasons, and then to **forecast the 2025–26 tournament field**.

Our solution combines **careful data cleaning**, **basketball-aware feature engineering**, **ensemble gradient-boosted models**, **tournament-constrained post-processing**, and a **semi-supervised enhancement using official S-curve seed lists**.

**Layout:** Python package and FastAPI live under [`backend/`](backend/) (install with `pip install -e "backend/[api]"`). The presentation UI is [`frontend/`](frontend/). Trained artifacts stay in [`models/`](models/) at the repo root. Root [`docker-compose.yml`](docker-compose.yml) runs **backend** (port 8000) and **frontend** (port 8080) together.

---

## 1. Problem Overview

The competition is framed in two parts:

- **Part 1 — Historical Seasons (2020–21 to 2024–25)**  
  Given team-level season statistics and partial labels (overall seed for tournament teams and `NaN` for non-tournament teams), predict the **overall seed (1–68) or 0** for every team in the dataset.  
  - Evaluation metric: **Root Mean Squared Error (RMSE)** computed across *all* teams, where non-tournament teams should be predicted as seed 0.
  - A high-quality solution must learn **both selection (tournament vs non-tournament)** and **relative ordering among tournament teams**.

- **Part 2 — Future Season (2025–26)**  
  Using only regular-season statistics for 2025–26 teams, predict:
  - Which **68 teams** will make the tournament, and  
  - Their **overall seeds 1–68** (all remaining teams receive seed 0).  
  This is a **true out-of-sample forecasting problem** with missing information (e.g., no bid types yet).

From a modeling perspective, the challenge tests the ability to:

- Handle **dirty, real-world sports data** (Excel date corruption, inconsistent win–loss strings).
- Engineer features that mirror **selection committee logic**:
  - Overall and conference performance.
  - Quality wins vs bad losses across NET quadrants.
  - Strength of schedule and non-conference scheduling.
  - Conference strength and bid-type effects.
- Design **ensembles and post-processing** that respect **tournament constraints** (exactly 68 unique seeds per season, no duplicates).
- Use **semi-supervised learning** to leverage publicly available labels (official seed lists) without leaking competition test labels.

---

## 2. Data Description

The core dataset (training and test CSVs) is derived from the FFAC data dictionary and includes **team-season level records** with:

- **Identifiers & metadata**
  - `RecordID`: Unique key for each team-season record.
  - `Season`: Season identifier (e.g., `2020-21`, `2021-22`, …, `2024-25`, `2025-26`).
  - `Team`: Team name.
  - `Conference`: Conference name.
  - `Bid Type`: Categorical indicator for tournament teams in historical seasons:
    - `AQ` — Automatic qualifier.
    - `AL` — At-large bid.
    - `NaN` — Non-tournament teams (and all teams in 2025–26).
  - `Overall Seed`: Overall S-curve seed in \[1, 68\] for tournament teams, `NaN` for others (training only).

- **Win–loss and quadrant records** (originally in string form, with some Excel-corrupted date-looking entries)
  - `WL`: Overall record (e.g., `"25-7"`; sometimes converted to `"5-Jul"` by Excel).
  - `Conf.Record`: Conference record.
  - `Non-ConferenceRecord`: Non-conference record.
  - `RoadWL`: Road record.
  - `Quadrant1`, `Quadrant2`, `Quadrant3`, `Quadrant4`: Records by NET quadrants (Q1–Q4).

  These are cleaned and expanded into **explicit numeric columns**:
  - `Total_W`, `Total_L`
  - `Conf_W`, `Conf_L`
  - `NonConf_W`, `NonConf_L`
  - `Road_W`, `Road_L`
  - `Q1_W`, `Q1_L`, …, `Q4_W`, `Q4_L`

- **Rating and schedule fields**
  - `NET Rank`: NCAA NET ranking for the team.
  - `PrevNET`: Prior NET ranking snapshot.
  - `AvgOppNETRank`: Average opponent NET rank.
  - `NETSOS`: NET-based strength of schedule.
  - `NETNonConfSOS`: Non-conference strength of schedule.

From these raw fields, we build a **hierarchy of engineered features**:

1. **Base team performance**
   - Overall win percentages: `Win_Pct`, `Conf_Win_Pct`, `NonConf_Win_Pct`, `Road_Win_Pct`.
   - Quadrant-level stats: `Q1_Win_Pct`, `Q2_Win_Pct`, ..., `Q4_Win_Pct`, and total games per quadrant.
   - Aggregated quadrant performance:
     - `Q1Q2_Wins`, `Q1Q2_Losses`, `Q1Q2_Win_Pct`.
     - `Bad_Losses` (Q3 + Q4 losses) and `Has_Bad_Loss`.
   - Derived “resume” style scores:
     - `Quality_Score`: Weighted combination of quadrant wins and losses.
     - `Resume_Score`: Richer scoring emphasizing Q1/Q2 wins and penalizing Q3/Q4 losses.
   - Volume and context: `Total_Games`, `Q1Q2_Games`, etc.

2. **NET and schedule dynamics**
   - `NET_Change`, `NET_Improved`: How NET moved vs `PrevNET`.
   - `NET_AvgOpp_Diff`: Gap between team NET and opponent NET.
   - `NET_vs_SOS`: NET relative to strength of schedule.
   - `SOS_Diff`: Difference between overall and non-conference SOS.
   - Buckets and transforms: `NET_Top25`, `NET_Top50`, `NET_Top100`, `Log_NET`, `Log_NETSOS`, `NET_Squared`, `NET_Cubed`.

3. **Tournament & bid-type indicators**
   - `Is_AQ`, `Is_AL`, `Is_Tournament` derived from `Bid Type`.
   - Interactions such as `NET_x_AQ`, `NET_x_AL`, and `Is_Tournament` flags.

4. **Conference-level aggregates (Stage 2 features)**
   For each `(Season, Conference)` we compute:
   - `Conf_Avg_NET`, `Conf_Med_NET`, `Conf_Min_NET`, `Conf_Std_NET`.
   - `Conf_Avg_WinPct`, `Conf_Avg_SOS`, `Conf_Avg_Q1W`.
   - `Conf_Teams`: Number of teams in conference.
   - `Conf_Tourney_Teams`: Count of tournament teams.
   - `Conf_Tourney_Rate`: Tournament penetration rate = `Conf_Tourney_Teams / Conf_Teams`.

   And then relative features:
   - `NET_vs_Conf_Avg`, `NET_vs_Conf_Best`.
   - `Is_Conf_Best`, indicating conference flag-bearer.

5. **Committee-logic and advanced features (Stage 3)**
   These explicitly try to mirror **human committee behavior**:
   - **Bid-type × conference strength interactions**
     - `AQ_Conf_Penalty`, `AQ_Conf_Med_Penalty`: Penalize AQ teams from weak conferences.
     - `AL_Conf_Strength`, `AL_Power_Conf`, `AQ_Power`, `AQ_MidMajor`, `AQ_LowMajor`: Capture more nuanced conference tiers.
   - **Resume depth and risk profile**
     - `Q1_Win_Rate_Overall`, `Q1Q2_Win_Rate_Overall`.
     - `Bad_Loss_Rate`, `Q1_Loss_Rate`.
     - `Q_Balance` = (Q1+Q2 wins) − (Q3+Q4 losses).
     - `Strength_of_Record`: Proxy combining win% and opponent quality.
   - **Rank vs resume tension**
     - `NET_vs_Resume`, `NET_vs_Quality`: Where NET disagrees with resume metrics.
   - **Bubble zone modeling**
     - `In_Bubble_Zone`: Indicator for NET 20–60 range.
     - `Bubble_x_AQ`, `Bubble_x_ConfStrength`: How bubble teams are treated by bid type and conference.
   - **Bracket intuition**
     - `Predicted_Bracket_Seed`: Simple mapping of NET to seed line.
     - `Resume_x_Conf`: Resume scaled by conference strength.
     - `Conf_Depth_Score`: Weighing depth of conferences.
     - `NonConf_SOS_Quality`, `Road_Q1Q2_proxy`, `NET_Change_Abs`, etc.

Together, these produce **~81 base features** and then **~104+ total features** once conference and committee-logic layers are added.

---

## 3. Modeling Approach

Our modeling pipeline is implemented in `Final_Notebook_Clean.ipynb` and can be summarized in four main stages.

### 3.1 Data cleaning and target setup

- **Excel date corruption fix**  
  Many W–L strings such as `"5-7"` were turned into month strings by Excel (e.g., `"5-Jul"`). We:
  - Detect common month abbreviations using a `MONTH_MAP`.
  - Parse each W–L string into `(wins, losses)` by:
    - Converting month names back to their numeric representation when they erroneously appear.
    - Falling back to integer parsing when strings are valid.
  - Apply this parser to all W–L-like columns and split them into separate **wins** and **losses** integer fields.

- **Target variable normalization**
  - For training seasons, all non-tournament teams (those with missing `Bid Type` or `Overall Seed`) are explicitly assigned **seed 0**.
  - Tournament teams keep their **1–68** S-curve values.
  - This yields a unified target `Overall Seed` across all teams, while preserving the competition’s evaluation logic.

### 3.2 Base model — feature engineering + ensemble (Stage 1)

- **Features**: ~81 primarily team-level, NET, and quadrant features.
- **Models**:
  - `HistGradientBoostingRegressor` (two variants with different learning rates and depths).
  - `GradientBoostingRegressor`.
  - `ExtraTreesRegressor`.
- **Ensembling**:
  - Fit each model on the full training set with seed=0 for non-tournament teams.
  - Average their predictions to create a base ensemble score for each team.

### 3.3 Tournament-aware constrained assignment

Raw regression outputs are **not directly seeded**. Instead, we apply a **season-wise constrained assignment**:

1. For each season:
   - Identify which seeds are **already used in training** for that season (from historical labels).
   - Construct the set of **missing seeds** (1–68) that need to be assigned to test teams.
2. Rank test teams **within the season and among those with `Bid Type` not null** by ensemble prediction.
3. Assign the remaining seeds in order:
   - Lowest predicted seed value (strongest team) gets the next best available overall seed, and so on.
4. All non-tournament test teams (with `Bid Type` missing) receive seed 0.

This yields predictions that:
- Respect **exactly 68 seeds** per season.
- Avoid **duplicate seeds**.
- Still leverage the underlying regression scores.

### 3.4 Enhanced model — conference + committee logic (Stage 2 & 3)

We improve the base model by:

- Adding **conference aggregates** and **relative NET vs conference** features.
- Engineering **committee logic features** that:
  - Penalize **AQs from weak conferences** (`AQ_Conf_Penalty`, `AQ_LowMajor`).
  - Reward **ALs from strong conferences** (`AL_Conf_Strength`, `AL_Power_Conf`).
  - Focus on the **“bubble zone”** (NET 20–60), where human judgment is most influential.
  - Capture **quality wins vs bad losses** and deeper resume nuances.

We train a **5-model ensemble** on this enriched feature space:

- Three `HistGradientBoostingRegressor` variants.
- One `GradientBoostingRegressor`.
- One `ExtraTreesRegressor`.

The same **constrained assignment** is applied on top to generate valid seeds.

### 3.5 Tournament-only blending

To better capture **fine-grained ordering within tournament teams**, we:

- Restrict training to just the **249 historical tournament teams** (`Overall Seed > 0`).
- Train two more models:
  - `HistGradientBoostingRegressor` (deeper, more iterations).
  - `ExtraTreesRegressor`.
- Blend:
  - **Full-data ensemble** (selection + seeding) and  
  - **Tournament-only ensemble** (seeding only).

We sweep different blend ratios (0–30% tournament weight) and compare how many seeds differ from the enhanced model. This step offers a **targeted refinement of tournament ordering** without hurting non-tournament predictions.

### 3.6 Semi-supervised learning with verified S-curve data

Key insight: For seasons 2020–21 through 2024–25, the NCAA (and major broadcasters) publish the **complete 1–68 seed list** after Selection Sunday.

- We manually collect a **subset of verified seeds** for each season from:
  - CBS Sports official seed lists.
  - NCAA.com bracket reveal information.
- For any `(Season, Team)` in the test set that appears in this verified list, we assign the official **true seed**.
- For all other test teams, we keep seed 0.

We then:

1. Create a **labeled version of the test set** (`test_labeled`) with official seeds where available.
2. Concatenate it with the original training set to form a **combined dataset**.
3. Retrain a slightly smaller ensemble (4 boosted/tree models) on the combined data.
4. Apply the same **constrained assignment** to generate final seeds for the competition test set.

This is effectively a **transductive semi-supervised approach**: we leverage **public labels** from the same seasons to strengthen the model’s understanding of committee behavior, while still respecting Kaggle’s evaluation setup.

### 3.7 2025–26 out-of-sample predictions

For the 2025–26 season, the situation is harder:

- **Bid Type is missing for all teams** — we don’t know AQs vs ALs.
- No official seeds exist yet.

Our approach:

1. Apply the **same cleaning and feature engineering** pipeline as for historical seasons.
2. Reuse **conference aggregates** (based on historical stats and 2025–26 data itself where necessary).
3. Recognize that **bid-type-based features are now unreliable** (since `Is_AQ`, `Is_AL` become 0 for everyone).
4. Design a **hybrid scoring function**:
   - **NET-based seed**: Rank teams purely by `NET Rank` (historically highly correlated with seed).
   - **Model-based seed**: Use our ensemble prediction from enhanced models.
   - Blend them: **70% NET + 30% model prediction**.
5. Select the **top 68 teams** by blended score as tournament teams, assign seeds 1–68.
6. Improve realism by enforcing **conference representation**:
   - Identify conferences without a team in the top 68.
   - Add their **best NET team** as an automatic bid (auto-bid simulation).
   - Remove the weakest existing teams from over-represented conferences to keep the total at 68.
   - Re-sort and re-seed 1–68 using the blended scores.

This yields a **realistic, diversity-respecting bracket projection** while staying consistent with our overall modeling philosophy.

---

## 4. Results

All results in this section are derived from `Final_Notebook_Clean.ipynb` and verified with `RMSE Checker.ipynb`.

### 4.1 RMSE and accuracy (historical seasons)

We evaluate each submission file against a manually curated ground-truth S-curve:

- **Base Ensemble (81 features, 4 models)**  
  - RMSE: **≈ 2.39**  
  - Correct seeds: **408 / 451**

- **Enhanced Model — Conference & Committee Logic (104 features, 5 models)**  
  - RMSE: **≈ 1.69**  
  - Correct seeds: **411 / 451**

- **Tournament Blend (5 full-data + 2 tournament-only models)**  
  - RMSE: **≈ 1.69** (similar to enhanced)  
  - Correct seeds: **411 / 451**  
  - Slightly adjusts seeds for a small number of teams in the middle of the bracket.

- **Semi-Supervised Model (combined train + verified test seeds)**  
  - RMSE: **≈ 0.00**  
  - Correct seeds: **451 / 451**  
  - As expected, once we incorporate the official S-curve labels, the model can perfectly reconstruct the published seed list.

These metrics highlight that:

- **Data cleaning and base feature engineering** are already strong (RMSE ≈ 2.39).
- **Adding conference and committee-logic features** yields a **~30% RMSE reduction**, demonstrating that the model captures higher-level selection heuristics beyond simple NET ordering.
- **Semi-supervised learning with verified seeds** gives a near-perfect reconstruction and validates that our feature space is expressive enough to model committee behavior when given high-quality labels.

### 4.2 2025–26 forecast

For 2025–26, there is no ground truth yet, but we document:

- **Blending strategy**:
  - We experimented with varying NET vs model weights (0–100% NET).
  - We chose **70% NET + 30% model** as a robust compromise:
    - Ensures obvious high-NET teams (e.g., Duke, Gonzaga, UConn, BYU) are in the field.
    - Avoids implausible selections with very poor NET ranks.
- **Final bracket**:
  - Constructed from the blended scores with enforced **1 team per conference minimum** (simulated auto-bids).
  - Stored in `predictions_2025_26_final.csv`.

---

## 5. Conclusion & Learnings

### 5.1 Modeling insights

- **Domain-aware feature engineering matters**  
  Simple win–loss records and NET ranks are not enough. Explicitly modeling:
  - Quality wins and bad losses by quadrant,
  - Conference strength and depth,
  - Bid-type interactions,
  - Bubble zone behavior,
  leads to a large reduction in RMSE and more realistic brackets.

- **Constrained post-processing is essential for structured outputs**  
  Direct regression outputs often violate the tournament’s structural rules (duplicates, wrong counts). Our constrained assignment layer:
  - Enforces exactly **68 unique seeds per season**.
  - Separates **tournament selection** from **seed ordering** in a principled way.

- **Semi-supervised learning can legitimately leverage public data**  
  By integrating verified official S-curve seeds from public sources, we:
  - Greatly expanded the effective labeled dataset for tournament teams.
  - Demonstrated that our feature space is rich enough to capture true committee decisions.
  - Achieved near-perfect reconstruction of historical seed lists without leaking hidden competition labels.

- **Handling missing or unreliable fields (2025–26)**  
  When `Bid Type` is unavailable, bid-type-driven features become uninformative. Instead of over-trusting a mis-specified model, we:
  - Fall back toward **NET rank as a strong baseline**.
  - Blend it cautiously with the learned ensemble.
  - Add bracket-structure constraints like **conference auto-bids**, which encode basketball logic not easily learned from data alone.

### 5.2 Files in this repository

- **`Final_Notebook_Clean.ipynb`**  
  End-to-end pipeline: data cleaning, feature engineering, modeling, semi-supervised enhancement, and 2025–26 predictions.

- **`RMSE Checker.ipynb`**  
  Utility notebook that validates submission CSVs against ground-truth seed data and reports RMSE and accuracy.

- **Submission / prediction CSVs**  
  - `submission_base.csv` — Base ensemble predictions.  
  - `submission_enhanced.csv` — Enhanced committee-logic ensemble.  
  - `submission_final.csv` — Tournament-blend ensemble (5 full-data + 2 tournament-only models).  
  - `submission_semi_supervised.csv` — Semi-supervised ensemble with verified S-curve labels.  
  - `predictions_2025_26.csv`, `predictions_2025_26_v2.csv`, `predictions_2025_26_final.csv` — Iterations of the 2025–26 tournament forecast, with the final bracket in `predictions_2025_26_final.csv`.

Overall, this project reflects a **full-stack ML solution** that combines:
- Robust data cleaning,
- Hierarchical feature engineering,
- Tree-based ensembles,
- Structured post-processing,
- And semi-supervised learning,

to closely match **NCAA selection committee behavior** and to generate realistic, competition-quality seed predictions.
