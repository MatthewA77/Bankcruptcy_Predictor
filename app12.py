import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from pygooglenews import GoogleNews
from newspaper import Article
import os
import nltk
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Bankruptcy Risk Predictor", layout="wide")

# --- Download NLTK data for newspaper library ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Load Pre-trained Model, Scaler, Feature List, and SHAP Explainer ---
@st.cache_resource
def load_model():
    try:
        models_path = 'models/'
        model = joblib.load(os.path.join(models_path, 'random_forest_model.joblib'))
        scaler = joblib.load(os.path.join(models_path, 'scaler.joblib'))
        top_10_features = joblib.load(os.path.join(models_path, 'top_10_features.joblib'))
        explainer = joblib.load(os.path.join(models_path, 'shap_explainer.joblib'))
        return model, scaler, top_10_features, explainer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please run the `model_taiwan.ipynb` notebook to generate all model assets.")
        return None, None, None, None

model, scaler, top_10_features, explainer = load_model()


# --- Feature Definitions (remains unchanged) ---
FEATURE_DEFINITIONS = {
    "Net Income to Stockholder's Equity": "Also known as Return on Equity (ROE). Measures a company's profitability by revealing how much profit a company generates with the money shareholders have invested.",
    "Net Value Growth Rate": "Shows the percentage increase or decrease in a company's net worth (Assets - Liabilities) from one period to the next. A higher rate indicates a growing company.",
    "Persistent EPS in the Last Four Seasons": "The average Earnings Per Share (EPS) over the past four quarters. It indicates the consistency and stability of a company's profitability.",
    "Borrowing dependency": "Calculated as Total Liabilities / Total Assets. This ratio shows the extent to which a company relies on debt to finance its assets. A high ratio can indicate high risk.",
    "Per Share Net profit before tax": "The company's profit before taxes, divided by the number of outstanding shares. It shows profitability on a per-share basis. (Note: The currency is based on the original dataset, but the calculation is universal).",
    "Total debt/Total net worth": "A leverage ratio that compares a company's total debt to its total net worth. It measures how much debt is used to finance the company's assets relative to the value owned by shareholders.",
    "Net Value Per Share (A)": "The company's net worth (Assets - Liabilities) divided by the number of outstanding shares. It represents the intrinsic value of a single share.",
    "Net Income to Total Assets": "Also known as Return on Assets (ROA). This ratio indicates how profitable a company is in relation to its total assets. It measures how efficiently a company is using its assets to generate earnings.",
    "Degree of Financial Leverage (DFL)": "Measures the sensitivity of a company's earnings per share to fluctuations in its operating income, as a result of changes in its capital structure. A high DFL means a small change in operating income will lead to a large change in earnings.",
    "Interest Expense Ratio": "Calculated as Interest Expense / Total Revenue. This ratio shows the proportion of a company's revenue that is used to pay the interest on its debt."
}


# --- Helper Functions (all remain unchanged) ---
@st.cache_data(ttl=3600)
def get_financial_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if not info or info.get('regularMarketPrice') is None:
             raise ValueError(f"Ticker '{ticker_symbol}' not found or no data available.")
        income_stmt = ticker.financials
        balance_sheet = ticker.balance_sheet
        quarterly_earnings = ticker.quarterly_earnings
        if income_stmt.empty or balance_sheet.empty:
            return None, None, None, None
        return info, income_stmt, balance_sheet, quarterly_earnings
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker_symbol}. Error: {e}")
        return None, None, None, None

@st.cache_data(ttl=3600)
def get_company_specific_news(query):
    try:
        gn = GoogleNews(lang='en', country='US')
        search_results = gn.search(query)
        articles_with_details = []
        for entry in search_results['entries'][:5]:
            try:
                article = Article(entry.link)
                article.download()
                article.parse()
                article.nlp()
                articles_with_details.append({
                    'title': entry.title,
                    'publisher': entry.source['title'],
                    'link': entry.link,
                    'published': entry.published,
                    'summary': article.summary,
                })
            except Exception:
                continue
        return articles_with_details
    except Exception as e:
        st.warning(f"Could not fetch news for '{query}'. Error: {e}")
        return []

def get_top_executive(officers_list):
    if not officers_list or not isinstance(officers_list, list): return "N/A"
    title_priority = ['CEO', 'Chief Executive Officer', 'President Director', 'Co-Founder']
    for title in title_priority:
        for officer in officers_list:
            if 'title' in officer and title.lower() in officer['title'].lower():
                return officer.get('name', 'N/A')
    if officers_list: return officers_list[0].get('name', 'N/A')
    return "N/A"

def map_data_to_features_for_year(info, income_stmt, balance_sheet, quarterly_earnings, feature_list, year_col):
    year_col_loc = balance_sheet.columns.get_loc(year_col)
    prev_year_col = balance_sheet.columns[year_col_loc + 1] if year_col_loc + 1 < len(balance_sheet.columns) else None
    def get_value(df, key, year):
        try:
            if key in df.index and year in df.columns: return df.loc[key, year]
            return 0
        except (KeyError, IndexError): return 0
    total_assets = get_value(balance_sheet, 'Total Assets', year_col)
    total_liabilities = get_value(balance_sheet, 'Total Liab', year_col)
    if total_liabilities == 0: total_liabilities = get_value(balance_sheet, 'Total Liabilities Net Minority Interest', year_col)
    stockholders_equity = get_value(balance_sheet, 'Stockholders Equity', year_col)
    if stockholders_equity == 0: stockholders_equity = get_value(balance_sheet, 'Total Stockholder Equity', year_col)
    total_debt = get_value(balance_sheet, 'Total Debt', year_col)
    net_worth = total_assets - total_liabilities
    net_income = get_value(income_stmt, 'Net Income', year_col)
    ebt = get_value(income_stmt, 'Pretax Income', year_col)
    if ebt == 0: ebt = get_value(income_stmt, 'EBT', year_col)
    ebit = get_value(income_stmt, 'EBIT', year_col)
    interest_expense = get_value(income_stmt, 'Interest Expense', year_col)
    total_revenue = get_value(income_stmt, 'Total Revenue', year_col)
    shares_outstanding = info.get('sharesOutstanding', 0)
    net_income_to_equity = net_income / stockholders_equity if stockholders_equity else 0
    net_value_growth = 0
    if prev_year_col:
        prev_assets = get_value(balance_sheet, 'Total Assets', prev_year_col)
        prev_liabilities = get_value(balance_sheet, 'Total Liab', prev_year_col)
        prev_net_worth = prev_assets - prev_liabilities
        if prev_net_worth: net_value_growth = (net_worth - prev_net_worth) / abs(prev_net_worth)
    if quarterly_earnings is not None and not quarterly_earnings.empty: persistent_eps = quarterly_earnings['EPS'].mean()
    else: persistent_eps = info.get('trailingEps', 0)
    borrowing_dependency = total_liabilities / total_assets if total_assets else 0
    profit_before_tax_per_share = ebt / shares_outstanding if shares_outstanding else 0
    debt_to_net_worth = total_debt / net_worth if net_worth else 0
    net_value_per_share = net_worth / shares_outstanding if shares_outstanding else 0
    net_income_to_assets = net_income / total_assets if total_assets else 0
    dfl = 0
    if (ebit - interest_expense) != 0: dfl = ebit / (ebit - interest_expense)
    interest_expense_ratio = interest_expense / total_revenue if total_revenue else 0
    feature_mapping = {
        "Net Income to Stockholder's Equity": net_income_to_equity, 'Net Value Growth Rate': net_value_growth,
        'Persistent EPS in the Last Four Seasons': persistent_eps, 'Borrowing dependency': borrowing_dependency,
        'Per Share Net profit before tax (Yuan ¥)': profit_before_tax_per_share, 'Total debt/Total net worth': debt_to_net_worth,
        'Net Value Per Share (A)': net_value_per_share, 'Net Income to Total Assets': net_income_to_assets,
        'Degree of Financial Leverage (DFL)': dfl, 'Interest Expense Ratio': interest_expense_ratio
    }
    ordered_values = [feature_mapping.get(feat.strip()) for feat in feature_list]
    return pd.DataFrame([ordered_values], columns=feature_list)

# --- MODIFIED: format_value now handles percentages, currencies, and ratios ---
def format_value(value, feature_name, currency_symbol="$"):
    """Formats a number based on its type (percentage, currency, or ratio)."""
    # Check for percentage-based features
    if any(keyword in feature_name for keyword in ["Rate", "dependency", "Ratio", "Equity", "Assets"]):
        return f"{value:.2%}"
    
    # Check for currency-based features
    if "Per Share" in feature_name or "EPS" in feature_name:
        return f"{currency_symbol} {value:,.2f}"

    # Default to ratio/number formatting
    if abs(value) >= 1_000_000_000: return f"{value / 1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000: return f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000: return f"{value / 1_000:.2f}K"
    else: return f"{value:.2f}"


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

def predictor_page():
    st.title('Bankruptcy Risk Predictor')
    st.markdown("Enter a stock ticker symbol to fetch its latest financial data, relevant news, and predict its bankruptcy risk.")

    with st.form(key='ticker_form'):
        ticker_input = st.text_input("Ticker Symbol", placeholder="e.g., AAPL, GOOG, TSLA").upper()
        submit_button = st.form_submit_button(label='Analyze')

    st.markdown("---")

    if submit_button and ticker_input:
        if not all([model, scaler, top_10_features, explainer]):
            st.error("Model assets are not loaded. Please check the files and restart.")
            st.stop()
        
        with st.spinner(f"Analyzing financial data and news for {ticker_input}..."):
            info, income, balance, q_earnings = get_financial_data(ticker_input)
            
            if info is not None:
                st.subheader(f"Financial Analysis for {ticker_input}")
                
                latest_year_col = balance.columns[0]
                latest_feature_df = map_data_to_features_for_year(info, income, balance, q_earnings, top_10_features, latest_year_col)
                scaled_features = scaler.transform(latest_feature_df)
                prediction = model.predict(scaled_features)
                prediction_proba = model.predict_proba(scaled_features)
                risk_prob = prediction_proba[0][1]

                # if prediction[0] == 0: 
                #     st.success("Latest Prediction: **Financially Stable**")
                # else: 
                #     st.error("Latest Prediction: **At Risk of Bankruptcy**")
                
                st.metric(label="Latest Calculated Probability of Bankruptcy", value=f"{risk_prob:.2%}")
                
                historical_risk_data = []
                historical_features_data = []
                available_years = balance.columns[:min(5, len(balance.columns)-1)]
                
                for year_col in available_years:
                    feature_df_year = map_data_to_features_for_year(info, income, balance, q_earnings, top_10_features, year_col)
                    
                    scaled_features_year = scaler.transform(feature_df_year)
                    risk_prob_year = model.predict_proba(scaled_features_year)[0][1]
                    historical_risk_data.append({'Year': year_col.year, 'Risk Probability': risk_prob_year})

                    feature_values_year = feature_df_year.iloc[0].to_dict()
                    feature_values_year['Year'] = year_col.year
                    historical_features_data.append(feature_values_year)
                
                risk_df = pd.DataFrame(historical_risk_data).sort_values(by='Year').reset_index(drop=True)
                st.subheader("Bankruptcy Risk Trend Over Time")
                st.line_chart(risk_df.rename(columns={'Year':'index'}).set_index('index'))
                st.info("The chart above shows the calculated bankruptcy risk probability over the last few years.")

                with st.expander("View Historical Feature Trends"):
                    if historical_features_data:
                        features_df = pd.DataFrame(historical_features_data)
                        features_df = features_df.sort_values(by='Year').set_index('Year')
                        
                        fig = go.Figure()
                        for feature in features_df.columns:
                            clean_name = feature.strip().replace(" (Yuan ¥)", "")
                            fig.add_trace(go.Scatter(x=features_df.index, y=features_df[feature],
                                                     mode='lines+markers',
                                                     name=clean_name))
                        
                        fig.update_layout(
                            title="Historical Trends of Top 10 Financial Ratios",
                            xaxis_title="Year",
                            yaxis_title="Value",
                            legend_title="Features"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("Click on feature names in the legend to hide or show their lines on the chart. Note that ratios have different scales.")
                    else:
                        st.warning("Could not generate historical feature data.")

                with st.expander("Company Details"):
                    st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
                    st.markdown(f"**Country:** {info.get('country', 'N/A')}")
                    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.markdown(f"**Website:** {info.get('website', 'N/A')}")
                    st.markdown(f"**CEO/President:** {get_top_executive(info.get('companyOfficers', []))}")
                    st.markdown(f"**Summary:** {info.get('longBusinessSummary', 'N/A')}")
                
                # --- MODIFIED: Display features in a formatted table ---
                with st.expander(f"View Top 10 Features (Latest Year)"):
                    features_to_display = latest_feature_df.to_dict(orient='records')[0]
                    currency_symbol = info.get('currency', '$') # Get currency, default to $
                    
                    table_data = []
                    for key in top_10_features:
                        display_name = key.strip().replace(" (Yuan ¥)", "")
                        value = features_to_display[key]
                        formatted_value = format_value(value, display_name, currency_symbol)
                        table_data.append({"Feature": display_name, "Value": formatted_value})
                    
                    features_table_df = pd.DataFrame(table_data)
                    st.dataframe(features_table_df, hide_index=True, use_container_width=True)

                
                with st.expander("View Prediction Breakdown", expanded=True):
                    st.write("This chart explains the model's prediction for the most recent year.")
                    
                    shap_values = explainer.shap_values(scaled_features)
                    
                    p = shap.force_plot(
                        explainer.expected_value[1],
                        shap_values[0, :, 1],
                        latest_feature_df.iloc[0],
                        matplotlib=False
                    )
                    st_shap(p)

                    st.markdown("---") 
                    
                    shap_values_for_bankruptcy = shap_values[0, :, 1]
                    feature_names = latest_feature_df.columns
                    
                    feature_impacts = list(zip(feature_names, shap_values_for_bankruptcy))
                    
                    red_features = [f"- {name.strip().replace(' (Yuan ¥)', '')}" for name, impact in feature_impacts if impact > 0]
                    blue_features = [f"- {name.strip().replace(' (Yuan ¥)', '')}" for name, impact in feature_impacts if impact < 0]

                    col1_impact, col2_impact = st.columns(2)

                    with col1_impact:
                        st.markdown("🔴 **Features Increasing Bankruptcy Risk:**")
                        if red_features:
                            st.markdown("\n".join(red_features))
                        else:
                            st.markdown("_None_")
                    
                    with col2_impact:
                        st.markdown("🔵 **Features Decreasing Bankruptcy Risk:**")
                        if blue_features:
                            st.markdown("\n".join(blue_features))
                        else:
                            st.markdown("_None_")


                st.markdown("---")
                st.subheader("Latest Company News")
                
                company_name = info.get('shortName', ticker_input)
                news_articles = get_company_specific_news(f"{company_name} {ticker_input}")
                if news_articles:
                    for article in news_articles:
                        st.markdown(f"**[{article['title']}]({article['link']})**")
                        st.markdown(f"<small>Published by: {article['publisher']}</small>", unsafe_allow_html=True)
                        if article['summary']:
                            st.write(article['summary'])
                        st.markdown("---")
                else:
                    st.warning(f"No recent news articles found for {company_name}.")

            else:
                st.warning("Could not retrieve or process data for the given ticker.")

def model_info_page():
    """Contains the UI and content for the Model Information page."""
    st.title("Model Information: Random Forest Classifier")
    
    st.header("What is a Random Forest?")
    st.markdown("""
    A Random Forest is a popular machine learning algorithm that belongs to the category of **ensemble learning**. The core idea is simple yet powerful: **the wisdom of the crowd is better than the wisdom of an individual.**

    Instead of relying on a single, complex decision tree, a Random Forest builds many small, simple decision trees during training. Each tree is trained on a random subset of the data and a random subset of the features.
    
    When it's time to make a prediction, each individual tree "votes" on the outcome. The Random Forest then chooses the prediction that receives the most votes. This process, known as **bagging** (Bootstrap Aggregating), helps to reduce overfitting and generally results in a more accurate and stable model.
    """)

    st.header("Dataset Used")
    st.markdown("""
    The model was trained on the **Taiwanese Economic Journal (TEJ) database** for the years 1999-2009. This is a well-known public dataset used for bankruptcy prediction research. 
    
    - **Content:** It contains financial ratios and indicators from thousands of Taiwanese companies.
    - **Target Variable:** A binary class indicating whether a company went bankrupt or not.
    - **Features:** The model uses the top 10 most predictive financial ratios from the dataset, as determined through feature selection techniques during the model development phase.
    """)

    st.header("Model Performance")
    st.markdown("Below are the performance metrics from the model's test set.")
    
    st.subheader("Classification Report")
    
    report_data = {
        'precision': [0.98, 0.35, None, 0.67, 0.94],
        'recall': [0.97, 0.55, None, 0.76, 0.97],
        'f1-score': [0.98, 0.43, 0.95, 0.70, 0.95],
        'support': [1320, 44, 1364, 1364, 1364]
    }
    report_index = ['Stable', 'Bankrupt', 'accuracy', 'macro avg', 'weighted avg']
    report_df = pd.DataFrame(report_data, index=report_index)
    
    st.dataframe(report_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
        'support': '{:,.0f}'
    }).highlight_null(color='transparent'))


    st.subheader("Confusion Matrix")
    st.image("plots/confusion_matrix_updated.png", 
             caption="Confusion Matrix: Shows True Positives, True Negatives, False Positives, and False Negatives.")

    st.subheader("Feature Importance")
    st.image("plots/feature_importance_adjusted.png",
             caption="Top 10 features used by the Random Forest model.")
    
    st.markdown("---")
    st.info(
        "**Disclaimer:** Model performance is based on a specific dataset and test set. Real-world performance may vary."
    )

def feature_definitions_page():
    """Contains the UI and content for the Feature Definitions page."""
    st.title("Feature Definitions")
    st.markdown("This page provides definitions for the top 10 financial ratios used by the prediction model.")
    st.markdown("---")

    if top_10_features:
        for feature in top_10_features:
            clean_feature = feature.strip().replace(" (Yuan ¥)", "")
            
            if clean_feature in FEATURE_DEFINITIONS:
                st.markdown(f'<h4><u>{clean_feature}</u></h4>', unsafe_allow_html=True)
                st.write(FEATURE_DEFINITIONS[clean_feature])
    else:
        st.warning("Feature list could not be loaded.")


# --- Main App Navigation ---


# --- Main App Navigation (Tabs) ---
def landing_page():
    st.title("Welcome to Bankruptcy Risk Predictor")
    st.markdown("""
    <style>
    .big-font { font-size:2.0em !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Predict bankruptcy risk for any company using advanced machine learning and financial data.</p>', unsafe_allow_html=True)
    # st.image("https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=800&q=80", caption="Financial Risk Analysis", use_container_width=True)
    st.markdown("""
    #### What can you do here?
    - Analyze bankruptcy risk for any company by ticker symbol
    - Explore model details and feature definitions
    - Visualize financial trends and prediction breakdowns
    
    **Get started by selecting a tab above!**
    """)
    st.info("This app is for educational purposes and should not be considered financial advice.")

tabs = st.tabs(["Home", "Bankruptcy Risk Predictor", "Model Information", "Feature Definitions"])

with tabs[0]:
    landing_page()
with tabs[1]:
    predictor_page()
with tabs[2]:
    model_info_page()
with tabs[3]:
    feature_definitions_page()

# Sidebar info only (no navigation)
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses a machine learning model to predict bankruptcy risk based on financial data. It should not be considered financial advice."
)
