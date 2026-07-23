# Content Monetization Modeler

A Machine Learning application that predicts YouTube ad revenue using video performance metrics, audience engagement, content category, device type, and viewer location.

🔗 **Live Demo:** 
```bash
https://content-monetization-modeler-bleh.streamlit.app/
```

---

## Project Overview

This project aims to estimate YouTube ad revenue using regression models and identify the factors that influence monetization performance.

The workflow includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Evaluation
- Hyperparameter Tuning
- Streamlit Deployment

---

##  Dataset

- **Rows:** 122,400
- **Columns:** 12
- **Target Variable:** `ad_revenue_usd`

### Features

- Views
- Likes
- Comments
- Watch Time
- Video Length
- Subscribers
- Category
- Device
- Country

---

##  Feature Engineering

Created additional features such as:

```python
engagement_rate = (likes + comments) / views

likes_per_view = likes / views

comments_per_view = comments / views

watch_time_per_view = watch_time_minutes / views
```

---

## Models Evaluated

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor

### Evaluation Metrics

- R² Score
- MAE
- RMSE

---

## Best Model

**Linear Regression**

| Metric | Score |
|----------|----------|
| R² Score | 0.9505 |
| MAE | 3.25 |
| RMSE | 13.78 |

Linear Regression outperformed all other models, indicating a strong linear relationship between watch time and ad revenue.

---

## Key Insights

- Watch Time was the strongest predictor of ad revenue.
- Audience engagement positively influenced revenue.
- Approximately 95% of revenue variation was explained by the model.
- Viewer retention had a greater impact than video length.

---

## Streamlit Application

The application allows users to:

- Enter video performance metrics
- Predict YouTube ad revenue
- View model performance metrics
- Explore project insights

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Seaborn

---
