import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

#load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\hp\Downloads\European_Bank.csv")

    df = df.drop(columns=['Year','Surname','CustomerId'])

    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0,29,45,60,100],
        labels=['<30','30-45','46-60','60+'],
        include_lowest=True
    )

    df['CreditBand'] = pd.cut(
        df['CreditScore'],
        bins=[300, 579, 719, 850],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    df['TenureGroup'] = pd.cut(
        df['Tenure'],
        bins=[-1, 2, 6, 10],
        labels=['New', 'Mid-term', 'Long-term']
    )

    non_zero_median = df.loc[df['Balance'] > 0, 'Balance'].median()
    def balance_segment(balance):
        if balance == 0:
            return 'Zero'
        elif balance <= non_zero_median:
            return 'Low'
        else:
            return 'High'
    df['BalanceSegment'] = df['Balance'].apply(balance_segment)
    
    high_value_threshold = df['Balance'].quantile(0.75)
    df['HighValueFlag'] = df['Balance'] > high_value_threshold
    
    return df

df = load_data()