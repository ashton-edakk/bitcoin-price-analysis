# Relationships Between Cryptocurrency Prices and Regulation Changes

## Project Overview
This project analyzes the relationship between **Bitcoin’s daily price movements** and **cryptocurrency regulation-related news**. Using historical Bitcoin price data and scraped news articles, we explore whether regulatory events are associated with changes in returns, volatility, and short-term market behavior.

The project combines **data collection, web scraping, data cleaning, exploratory data analysis, visualization, and machine learning** to study how regulatory news correlates with Bitcoin price behavior over time.

---

## Data Sources
- **Bitcoin Price Data**
  - Historical daily BTC price data (open, high, low, close)

- **News Data**
  - Regulation-related cryptocurrency articles scraped from CoinDesk
  - Articles collected via the Internet Archive (Wayback Machine)
  - Articles labeled by event type using keyword-based classification

---

## Methodology

### Data Collection
- Pulled historical Bitcoin price data from Yahoo Finance
- Scraped CoinDesk regulation articles (2014–2020) using Wayback Machine snapshots
- Stored articles in structured CSV format

### Data Cleaning & Feature Engineering
- Aligned BTC price data and news articles by date
- Created binary and count-based indicators for news events
- Generated lag features (1–3 days after news)
- Computed daily percent change and price range

### Exploratory Data Analysis
- Compared return distributions on news vs. non-news days
- Analyzed volatility and extreme price movements
- Examined effects of article frequency and regulation intensity

### Visualization
- Time series plots of Bitcoin price with regulation days highlighted
- Distribution plots of daily returns
- Bar charts comparing average returns by event type
- Scatter plots showing news intensity vs. volatility
- Timeline visualizations of regulation coverage

---

## Machine Learning Models
The project explores simple predictive models to evaluate whether news features add predictive power:

- **Logistic Regression**
  - Predicts whether BTC price increases or decreases
  - Result: Low predictive accuracy due to market volatility

- **Linear Regression**
  - Predicts daily percent change
  - Shows limited contribution from news-related features

- **Random Forest Regressor**
  - Evaluates feature importance
  - Price-based features dominate predictive power

---

## Limitations
- News data limited to CoinDesk articles and 2014–2020
- Keyword-based labeling may miss nuance or sentiment
- Models are intentionally simple and not optimized for trading
- Results are correlational, not causal

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- yfinance
- BeautifulSoup
- Wayback Machine (Internet Archive)

---

## Disclaimer
This project is for **academic and educational purposes only**. It does not constitute financial advice or a trading strategy.
