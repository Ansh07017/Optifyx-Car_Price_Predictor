import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models(models, X_test, y_test, X_train, y_train, feature_names):
    """
    Evaluate the performance of trained models and visualize the results
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test target values
        X_train: Training features
        y_train: Training target values
        feature_names: Names of the features
    """
    st.write("## Model Evaluation")
    
    if not models:
        st.warning("No models to evaluate. Please train at least one model.")
        return
    
    # Create a dataframe to store evaluation metrics
    metrics_df = pd.DataFrame(
        columns=['Model', 'Train RMSE', 'Test RMSE', 'Test MAE', 'Test R²']
    )
    
    # Calculate evaluation metrics for each model
    for model_name, model_info in models.items():
        model = model_info['model']
        
        # Make predictions on train and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Add to metrics dataframe
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Model': [model_name],
            'Train RMSE': [train_rmse],
            'Test RMSE': [test_rmse],
            'Test MAE': [test_mae],
            'Test R²': [test_r2]
        })], ignore_index=True)
    
    # Display metrics
    st.write("### Model Performance Metrics")
    
    # Format the dataframe for display
    formatted_df = metrics_df.copy()
    for column in ['Train RMSE', 'Test RMSE', 'Test MAE']:
        formatted_df[column] = formatted_df[column].round(2)
    formatted_df['Test R²'] = formatted_df['Test R²'].round(3)
    
    st.dataframe(formatted_df)
    
    # Create bar chart for comparison
    st.write("### Comparison of Model RMSE (Lower is better)")
    fig = px.bar(
        metrics_df, 
        x='Model', 
        y='Test RMSE', 
        title='Test RMSE by Model',
        color='Test RMSE',
        labels={'Test RMSE': 'Root Mean Squared Error'},
        color_continuous_scale='reds_r'  # Reversed red scale so lower (better) values are lighter
    )
    st.plotly_chart(fig)
    
    # Create bar chart for R² (Higher is better)
    st.write("### Comparison of Model R² (Higher is better)")
    fig = px.bar(
        metrics_df, 
        x='Model', 
        y='Test R²', 
        title='Test R² by Model',
        color='Test R²',
        labels={'Test R²': 'R² Score'},
        color_continuous_scale='blues'  # Blue scale so higher (better) values are darker
    )
    st.plotly_chart(fig)
    
    # Find the best model based on Test RMSE
    best_model_idx = metrics_df['Test RMSE'].idxmin()
    best_model_name = metrics_df.loc[best_model_idx, 'Model']
    best_model = models[best_model_name]['model']
    
    st.write(f"### Best Performing Model: {best_model_name}")
    st.write(f"Test RMSE: {metrics_df.loc[best_model_idx, 'Test RMSE']:.2f}")
    st.write(f"Test R²: {metrics_df.loc[best_model_idx, 'Test R²']:.3f}")
    
    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        st.write("### Feature Importance Analysis")
        
        # Get feature importance
        importances = best_model.feature_importances_
        
        # Create dataframe for feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Display as a table
        st.dataframe(feature_importance_df)
        
        # Create bar chart for feature importance
        fig = px.bar(
            feature_importance_df,
            x='Feature',
            y='Importance',
            title=f'Feature Importance for {best_model_name}',
            color='Importance',
            labels={'Importance': 'Importance Score'},
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig)
    
    # Prediction vs Actual plot for the best model
    st.write("### Predictions vs Actual Values")
    
    # Make predictions with best model
    y_test_pred_best = best_model.predict(X_test)
    
    # Create dataframe for visualization
    pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred_best
    })
    
    # Create scatter plot with plotly
    fig = px.scatter(
        pred_df,
        x='Actual',
        y='Predicted',
        title='Actual vs Predicted Values',
        labels={'Actual': 'Actual Price', 'Predicted': 'Predicted Price'},
        opacity=0.7
    )
    
    # Add perfect prediction line
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='gray'),
        y0=pred_df['Actual'].min(),
        y1=pred_df['Actual'].max(),
        x0=pred_df['Actual'].min(),
        x1=pred_df['Actual'].max()
    )
    
    st.plotly_chart(fig)
    
    # Residuals analysis
    st.write("### Residuals Analysis")
    
    # Calculate residuals
    residuals = y_test - y_test_pred_best
    
    # Create dataframe for visualization
    residual_df = pd.DataFrame({
        'Predicted': y_test_pred_best,
        'Residual': residuals
    })
    
    # Create scatter plot for residuals
    fig = px.scatter(
        residual_df,
        x='Predicted',
        y='Residual',
        title='Residuals vs Predicted Values',
        labels={'Predicted': 'Predicted Price', 'Residual': 'Residual (Actual - Predicted)'},
        color='Residual',
        color_continuous_scale='RdBu_r',
        opacity=0.7
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='gray'),
        y0=0,
        y1=0,
        x0=residual_df['Predicted'].min(),
        x1=residual_df['Predicted'].max()
    )
    
    st.plotly_chart(fig)
    
    # Distribution of residuals
    fig = px.histogram(
        residual_df,
        x='Residual',
        title='Distribution of Residuals',
        labels={'Residual': 'Residual (Actual - Predicted)'},
        nbins=30,
        marginal='box'
    )
    
    st.plotly_chart(fig)
