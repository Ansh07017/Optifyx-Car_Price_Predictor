import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the car dataset
    
    Args:
        df (pandas.DataFrame): The car dataset
    """
    st.write("## Exploratory Data Analysis")
    
    # Sidebar for EDA options
    st.sidebar.subheader("EDA Options")
    eda_option = st.sidebar.selectbox(
        "Choose analysis type",
        ["Data Overview", "Univariate Analysis", "Bivariate Analysis", "Correlation Analysis"]
    )
    
    if eda_option == "Data Overview":
        st.write("### Dataset Overview")
        
        # Summary statistics
        st.write("#### Summary Statistics:")
        st.dataframe(df.describe())
        
        # Distribution of categorical variables
        st.write("#### Categorical Columns Distribution:")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            count_data = df[col].value_counts().reset_index()
            count_data.columns = [col, 'Count']
            
            # Use Plotly for interactive bar chart
            fig = px.bar(count_data, x=col, y='Count', title=f'Distribution of {col}')
            st.plotly_chart(fig)
    
    elif eda_option == "Univariate Analysis":
        st.write("### Univariate Analysis")
        
        # Select column for analysis
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        selected_col = st.selectbox("Select column for analysis:", numerical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            st.write(f"#### Histogram of {selected_col}")
            fig = px.histogram(df, x=selected_col, nbins=30, marginal="box", 
                               title=f"Distribution of {selected_col}")
            st.plotly_chart(fig)
        
        with col2:
            # Box plot
            st.write(f"#### Box Plot of {selected_col}")
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig)
        
        # Statistics
        st.write(f"#### Statistics for {selected_col}")
        stats = df[selected_col].describe()
        st.dataframe(stats)
        
        # Skewness and Kurtosis
        skewness = df[selected_col].skew()
        kurtosis = df[selected_col].kurt()
        
        st.write(f"**Skewness:** {skewness:.2f}")
        st.write(f"**Kurtosis:** {kurtosis:.2f}")
        
        if abs(skewness) > 1:
            st.write("*The distribution is highly skewed.*")
        elif abs(skewness) > 0.5:
            st.write("*The distribution is moderately skewed.*")
        else:
            st.write("*The distribution is approximately symmetric.*")
    
    elif eda_option == "Bivariate Analysis":
        st.write("### Bivariate Analysis")
        
        # Select columns for analysis
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        analysis_type = st.radio("Select analysis type:", ["Numerical vs Numerical", "Categorical vs Numerical"])
        
        if analysis_type == "Numerical vs Numerical":
            x_col = st.selectbox("Select X-axis column:", numerical_cols, index=0)
            y_col = st.selectbox("Select Y-axis column:", [col for col in numerical_cols if col != x_col], index=0)
            
            # Scatter plot
            st.write(f"#### Scatter Plot: {x_col} vs {y_col}")
            fig = px.scatter(df, x=x_col, y=y_col, 
                             title=f"{x_col} vs {y_col}", 
                             trendline="ols")
            st.plotly_chart(fig)
            
            # Correlation
            correlation = df[x_col].corr(df[y_col])
            st.write(f"**Correlation coefficient:** {correlation:.2f}")
            
            if abs(correlation) > 0.7:
                st.write("*Strong correlation*")
            elif abs(correlation) > 0.3:
                st.write("*Moderate correlation*")
            else:
                st.write("*Weak correlation*")
                
        elif analysis_type == "Categorical vs Numerical":
            cat_col = st.selectbox("Select categorical column:", categorical_cols, index=0)
            num_col = st.selectbox("Select numerical column:", numerical_cols, index=0)
            
            # Box plot
            st.write(f"#### Box Plot: {cat_col} vs {num_col}")
            
            # Limit categories for visualization if there are too many
            value_counts = df[cat_col].value_counts()
            if len(value_counts) > 10:
                top_categories = value_counts.nlargest(10).index.tolist()
                filtered_df = df[df[cat_col].isin(top_categories)].copy()
                st.write("*Showing only top 10 categories due to large number of categories*")
            else:
                filtered_df = df.copy()
            
            fig = px.box(filtered_df, x=cat_col, y=num_col, 
                        title=f"{cat_col} vs {num_col}")
            st.plotly_chart(fig)
            
            # Group statistics
            st.write(f"#### {cat_col} vs {num_col} Statistics")
            group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'count'])
            st.dataframe(group_stats.sort_values('mean', ascending=False))
            
            # Bar chart for mean values
            st.write(f"#### Mean {num_col} by {cat_col}")
            
            # Use the filtered dataframe for consistent visualization
            fig = px.bar(group_stats.reset_index(), x=cat_col, y='mean', 
                        title=f"Mean {num_col} by {cat_col}",
                        error_y='std')
            st.plotly_chart(fig)
    
    elif eda_option == "Correlation Analysis":
        st.write("### Correlation Analysis")
        
        # Select only numerical columns for correlation
        numerical_df = df.select_dtypes(include=['number'])
        
        # Correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Heatmap using Plotly
        fig = px.imshow(corr_matrix, 
                        text_auto='.2f', 
                        aspect="auto", 
                        title='Correlation Matrix',
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
        
        # Top correlations with Selling_Price
        if 'Selling_Price' in corr_matrix.columns:
            st.write("#### Top Correlations with Selling Price")
            price_corr = corr_matrix['Selling_Price'].drop('Selling_Price').sort_values(ascending=False)
            
            # Bar chart for correlations
            fig = px.bar(
                x=price_corr.index, 
                y=price_corr.values,
                title='Feature Correlation with Selling Price',
                labels={'x': 'Features', 'y': 'Correlation Coefficient'},
                color=price_corr.values,
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig)
            
            # Text explanation
            st.write("#### Interpretation")
            st.write("""
            - A positive correlation means as the feature increases, the selling price tends to increase.
            - A negative correlation means as the feature increases, the selling price tends to decrease.
            - Correlation strength: 
                - 0.0 to 0.3: weak
                - 0.3 to 0.7: moderate
                - 0.7 to 1.0: strong
            """)
