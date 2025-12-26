# College Dorm Review Sentiment Analysis & Satisfaction Modeling

Analyze student dorm reviews and rating sub-scores to understand what drives housing satisfaction, then test simple predictive + clustering baselines for pattern discovery.

> **TL;DR**: Built a Python NLP + analytics pipeline on ~1.3k dorm reviews. Created an n-gram–based sentiment score from free text, validated relationships with ratings using correlation + chi-square tests, and explored KNN regression + KMeans clustering.

---

## Project Goals
- Turn free-text dorm reviews into a **quantitative sentiment signal**
- Identify which rating sub-scores are associated with each other (e.g., bathroom vs building)
- Quantify how strongly text sentiment aligns with numeric ratings
- Explore lightweight baselines for:
  - **Prediction** (KNN regression)
  - **Segmentation** (KMeans clustering + PCA visualization)

---

## Data
- Student dorm reviews with numeric rating categories (e.g., Overall, Bathroom, Building, Location, Room) plus free-text comments.
- Dataset size: **~1,300+ reviews**

> If you’re open-sourcing this repo, don’t upload scraped raw data if terms prohibit it. Instead, include a small anonymized sample or instructions to reproduce the scrape.

---

## Approach (High Level)

### 1) Text Processing + Sentiment Feature
- Tokenized/cleaned review text
- Built an **n-gram sentiment score** from frequent positive/negative phrases
- Added `SentimentScore` as a feature alongside numeric rating columns

### 2) Statistical Validation
- **Pearson correlation** between sentiment and overall rating  
  - Example finding: `r ≈ 0.545` (highly significant)
- **Chi-square test** to evaluate association between rating distributions  
  - Example finding: `chi-squared(df=16) ≈ 696.9, p < 0.001`

### 3) Modeling + Clustering (Baselines)
- **KNN regression** to map ratings → sentiment (or related targets)
- **KMeans clustering** to segment dorm profiles
- **PCA** used for 2D visualization of clusters

---

## Results Snapshot
- Text sentiment and numeric ratings are strongly related:
  - **Pearson r ≈ 0.545 (p < 0.001)**
- Rating categories show statistically significant association:
  - **chi-squared(df=16) ≈ 696.9 (p < 0.001)**
- Baseline modeling/clustering demonstrates feasibility, with room for stronger ML baselines.

---

## Repo Structure (suggested)
