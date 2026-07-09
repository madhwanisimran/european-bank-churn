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
    df = pd.read_csv(r"European_Bank.csv")

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

#title & tabs
st.title("🏦 European Bank Customer Churn Analytics")
st.markdown("**Segmentation-driven churn analysis across 10,000 customers**")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🌍 Geographic & Demographic",
    "🔍 Segment Deep Dive",
    "💎 High Value Customers"
])

#TAB 1:Executive Overview
with tab1:
    st.header("Executive Overview")

    #KPI Cards
    total_customers = len(df)
    total_churned = df['Exited'].sum()
    total_retained = total_customers - total_churned
    churn_rate = df['Exited'].mean() * 100

    churned_hv = df[(df['HighValueFlag'] == True) & (df['Exited'] == 1)]
    revenue_at_risk = churned_hv['Balance'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{total_churned:,}")
    col3.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        churn_dist = df['Exited'].value_counts().reset_index()
        churn_dist.columns = ['Status', 'Count']
        churn_dist['Status'] = churn_dist['Status'].map({0: 'Retained', 1: 'Churned'})

        fig = px.bar(
            churn_dist,
            x='Status', y='Count',
            color='Status',
            color_discrete_map={'Retained': '#4E79A7', 'Churned': '#E15759'},
            text='Count',
            title='Retained vs Churned Customers'
        )

        fig.update_traces(textposition = 'outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Key Findings")
            st.markdown("""
        - 🔴 **Germany** has the highest churn rate at **32.44%** — double France & Spain
        - 🔴 **Age 46-60** group churns at **51.12%** — highest risk age segment  
        - 🔴 **Inactive members** drive **63.92%** of all churn
        - 🟡 Customers with **3-4 products** churn at **83-100%**
        - 🟢 **2-product customers** are the most loyal at **7.58%** churn
        - 🔴 **$88.6M** in high-value customer balances at risk
        """)
            
#TAB2 : geographic & demographic
with tab2:
    st.header("Geographic & Demographic Analysis")

    col1, col2 = st.columns(2)
    with col1:
        geo_filter = st.multiselect(
            "Filter by Geography",
            options=df['Geography'].unique(),
            default=df['Geography'].unique()
        )
        
    with col2:
        gender_filter = st.multiselect(
            "Filter by Gender",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )

    filtered_df = df[
        (df['Geography'].isin(geo_filter)) &
        (df['Gender'].isin(gender_filter)) 
    ]

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        geo_churn = filtered_df.groupby('Geography')['Exited'].mean().mul(100).round(2).reset_index()

        geo_churn.columns = ['Geography','Churn Rate (%)']

        fig = px.bar(
            geo_churn,
            x='Geography', y='Churn Rate (%)',
            color='Geography',
            color_discrete_map={'France': '#4E79A7', 'Germany': '#E15759', 'Spain': '#76B7B2'},
            text='Churn Rate (%)',
            title='Churn Rate by Geography'
        )

        fig.add_hline(y=20.37, line_dash='dash',
                      line_color='red', annotation_text = 'Baseline 20.37%')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_churn = filtered_df.groupby('Gender')['Exited'].mean().mul(100).round(2).reset_index()

        gender_churn.columns = ['Gender','Churn Rate (%)']

        fig = px.bar(
            gender_churn,
            x='Gender', y='Churn Rate (%)',
            color='Gender',
            color_discrete_map={'Female':'#B07AA1', 'Male':'#499894'},
            text='Churn Rate (%)',
            title='Churn Rate by Gender'
        )

        fig.add_hline(y=20.37, line_dash='dash',
                      line_color='red', annotation_text = 'Baseline 20.37%')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Geography x Age heatmap
    st.subheader("Geography × Age Group Interaction")

    geo_age = filtered_df.groupby(
        ['Geography', 'AgeGroup'], observed=True
    )['Exited'].mean().mul(100).round(2).unstack()

    geo_age = geo_age[['<30', '30-45', '46-60', '60+']]


    fig = px.imshow(
       geo_age,
       text_auto=True,
       color_continuous_scale='YlOrRd',
       aspect='auto',
       title='Churn Rate (%) — Geography × Age Group'
    )
    fig.update_layout(
       xaxis_title='Age Group',
       yaxis_title='Geography',
       coloraxis_colorbar=dict(title='Churn Rate (%)')
    )
    st.plotly_chart(fig, use_container_width=True)

#TAB3 : Segment Deep Dive
with tab3:
    st.header("Segment Deep Dive")
    col1, col2 = st.columns(2)

    with col1:
        age_filter = st.multiselect(
            "Filter by Age Group",
            options=['<30','30-45','46-60','60+'],
            default=['<30','30-45','46-60','60+']
        )

        with col2:
            balance_filter = st.multiselect(
                "Filter by Balance Segment", 
                options=['Zero','Low','High'],
                default=['Zero','Low','High']
            )

        seg_df = df[
            (df['AgeGroup'].isin(age_filter)) &
            (df['BalanceSegment'].isin(balance_filter)) 
        ]

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
           age_churn = seg_df.groupby('AgeGroup')['Exited'].mean().mul(100).round(2).reset_index()

           age_churn.columns = ['AgeGroup','Churn Rate (%)']
           age_churn['AgeGroup'] = age_churn['AgeGroup'].astype(str)

           fig = px.bar(
            age_churn,
            x='AgeGroup', y='Churn Rate (%)',
            color='AgeGroup',
            color_discrete_map={'<30': '#4E79A7',
            '30-45': '#F28E2B',
            '46-60': '#E15759',
            '60+': '#76B7B2'},
            text='Churn Rate (%)',
            title='Churn Rate by Age Group'
           )

           fig.add_hline(y=20.37, line_dash='dash',
                      line_color='red', annotation_text = 'Baseline 20.37%')
           fig.update_traces(textposition='outside')
           fig.update_layout(showlegend=False)
           st.plotly_chart(fig, use_container_width=True)

        with col2:
           product_churn = seg_df.groupby('NumOfProducts')['Exited'].mean().mul(100).round(2).reset_index()

           product_churn.columns = ['NumOfProducts','Churn Rate (%)']

           fig = px.bar(
            product_churn,
            x='NumOfProducts', y='Churn Rate (%)',
            color='NumOfProducts',
            color_discrete_map={'#59A14F', '#4E79A7', '#E15759', '#EDC948'},
            text='Churn Rate (%)',
            title='Churn Rate by Number of Products'
           )

           fig.add_hline(y=20.37, line_dash='dash',
                      line_color='red', annotation_text = 'Baseline 20.37%')
           fig.update_traces(textposition='outside')
           fig.update_layout(showlegend=False)
           st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            bal_churn = seg_df.groupby('BalanceSegment')['Exited'].mean().mul(100).round(2).reset_index()
            bal_churn.columns = ['BalanceSegment', 'Churn Rate (%)']

            bal_churn['BalanceSegment'] = bal_churn['BalanceSegment'].astype(str)
            bal_churn = bal_churn.set_index('BalanceSegment').reindex(['Zero', 'Low', 'High']).reset_index()
    


            fig = px.bar(
                bal_churn,
                x='BalanceSegment', y='Churn Rate (%)',
                color='BalanceSegment',
                color_discrete_map={'Zero': '#4E79A7', 'Low': '#F28E2B', 'High': '#E15759'},
                text='Churn Rate (%)',
                title='Churn Rate by Balance Segment'
            )

            fig.add_hline(y=20.37, line_dash='dash',
                          line_color='red', annotation_text = 'Baseline 20.37%')
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            with col2:
                active_churn = seg_df.groupby('IsActiveMember')['Exited'].mean().mul(100).round(2).reset_index()
                active_churn.columns=['IsActiveMember', 'Churn Rate (%)']
                active_churn['IsActiveMember'] = active_churn['IsActiveMember'].map({0:'Inactive', 1:'Active'})

                fig = px.bar(
                    active_churn,
                    x='IsActiveMember', y='Churn Rate (%)',
                    color='IsActiveMember',
                    color_discrete_map={'Inactive':'#E15759', 'Active':'#59A14F'},
                    text='Churn Rate (%)',
                    title='Churn Rate by Member Activity Status'
                )

                fig.add_hline(y=20.37, line_dash='dash',
                              line_color='red', annotation_text = 'Baseline 20.37%')
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Churned vs Retained - Average Profile Comparison")
            numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
            profile = seg_df.groupby('Exited')[numeric_cols].mean().round(2)
            profile.index = ['Retained', 'Churned']
            st.dataframe(profile, use_container_width=True)

#TAB4 : High Value Customers
with tab4:
    st.header("High Value Customer Analysis")
    
    geo_filter_hv = st.multiselect(
        "Filter by Geography",
        options=df['Geography'].unique(),
        default=df['Geography'].unique(),
        key='hv_geo'
    )
    
    hv_df = df[df['Geography'].isin(geo_filter_hv)]
    
    st.divider()
    
    # KPI Cards
    hv_customers = hv_df[hv_df['HighValueFlag'] == True]
    churned_hv = hv_customers[hv_customers['Exited'] == 1]
    revenue_at_risk = churned_hv['Balance'].sum()
    hv_churn_rate = hv_customers['Exited'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("High Value Customers", f"{len(hv_customers):,}")
    col2.metric("HV Churn Rate", f"{hv_churn_rate:.2f}%")
    col3.metric("Balance at Risk", f"${revenue_at_risk:,.0f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        hv_comparison = hv_df.groupby('HighValueFlag')['Exited'].mean().mul(100).round(2).reset_index()
        hv_comparison.columns = ['HighValueFlag', 'Churn Rate (%)']
        hv_comparison['Segment'] = hv_comparison['HighValueFlag'].map({False: 'Regular', True: 'High Value'})
        
        fig = px.bar(
            hv_comparison,
            x='Segment', y='Churn Rate (%)',
            color='Segment',
            color_discrete_map={'Regular': '#4E79A7', 'High Value': '#E15759'},
            text='Churn Rate (%)',
            title='Churn Rate — High Value vs Regular'
        )
        fig.add_hline(y=20.37, line_dash='dash', line_color='red',
                      annotation_text='Baseline 20.37%')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        hv_geo = hv_customers.groupby('Geography')['Exited'].mean().mul(100).round(2).reset_index()
        hv_geo.columns = ['Geography', 'Churn Rate (%)']
        
        fig = px.bar(
            hv_geo,
            x='Geography', y='Churn Rate (%)',
            color='Geography',
            color_discrete_sequence=['#4E79A7', '#EDC948', '#76B7B2'],
            text='Churn Rate (%)',
            title='High Value Churn Rate by Geography'
        )
        fig.add_hline(y=20.37, line_dash='dash', line_color='red',
                      annotation_text='Baseline 20.37%')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Financial profile comparison
    st.subheader("High Value — Churned vs Retained Financial Profile")
    hv_profile = hv_customers.groupby('Exited')[['Balance', 'EstimatedSalary']].mean().round(2)
    hv_profile.index = ['Retained', 'Churned']
    hv_profile = hv_profile.reset_index()
    hv_profile.columns = ['Status','Balance','EstimatedSalary']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            hv_profile,
            x='Status', y='Balance',
            color='Status',
            color_discrete_map={'Retained': '#59A14F', 'Churned': '#E15759'},
            text='Balance',
            title='Avg Balance — HV Churned vs Retained'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            hv_profile,
            x='Status', y='EstimatedSalary',
            color='Status',
            color_discrete_map={'Retained': '#59A14F', 'Churned': '#E15759'},
            text='EstimatedSalary',
            title='Avg Salary — HV Churned vs Retained'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    