import streamlit as st
import numpy as np
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

def train_models(X_train, y_train, feature_names, show_details=True):
    """
    Train multiple regression models on the car dataset
    
    Args:
        X_train: Training features
        y_train: Training target values
        feature_names: Names of the features
        show_details (bool): Whether to display training details in the UI
        
    Returns:
        dict: Trained models
    """
    # Only show UI elements if show_details is True
    if show_details:
        st.write("## Model Training")
        
        # Create a selection box for the models
        st.sidebar.subheader("Model Selection")
        models_to_train = st.sidebar.multiselect(
            "Select models to train:",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", 
             "Decision Tree", "Random Forest", "Gradient Boosting",
             "Support Vector Machine", "K-Nearest Neighbors"],
            default=["Linear Regression", "Random Forest", "Gradient Boosting"]
        )
        
        # Progress bar and status text for display
        progress_bar = st.progress(0)
        status_text = st.empty()
    else:
        # If not showing details, use default models
        models_to_train = ["Linear Regression", "Random Forest", "Gradient Boosting"]
        # Create empty placeholder objects that won't be displayed
        progress_bar = None
        status_text = None
    
    # Dictionary to store all trained models
    trained_models = {}
    
    # Setup model parameters based on selection
    models_config = {}
    
    # When show_details is False, use default parameters for selected models
    if not show_details:
        if "Linear Regression" in models_to_train:
            models_config["Linear Regression"] = LinearRegression()
        
        if "Random Forest" in models_to_train:
            models_config["Random Forest"] = RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42
            )
        
        if "Gradient Boosting" in models_to_train:
            models_config["Gradient Boosting"] = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
    else:
        # Interactive UI elements when show_details is True
        if "Linear Regression" in models_to_train:
            models_config["Linear Regression"] = LinearRegression()
        
        if "Ridge Regression" in models_to_train:
            alpha = st.sidebar.slider("Ridge alpha:", 0.01, 10.0, 1.0)
            models_config["Ridge Regression"] = Ridge(alpha=alpha)
        
        if "Lasso Regression" in models_to_train:
            alpha = st.sidebar.slider("Lasso alpha:", 0.01, 10.0, 1.0)
            models_config["Lasso Regression"] = Lasso(alpha=alpha)
        
        if "Decision Tree" in models_to_train:
            max_depth = st.sidebar.slider("Decision Tree max depth:", 2, 20, 5)
            models_config["Decision Tree"] = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        
        if "Random Forest" in models_to_train:
            n_estimators = st.sidebar.slider("Random Forest n_estimators:", 10, 200, 100)
            max_depth = st.sidebar.slider("Random Forest max depth:", 2, 20, 5)
            models_config["Random Forest"] = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        
        if "Gradient Boosting" in models_to_train:
            n_estimators = st.sidebar.slider("Gradient Boosting n_estimators:", 10, 200, 100)
            learning_rate = st.sidebar.slider("Gradient Boosting learning rate:", 0.01, 0.3, 0.1)
            models_config["Gradient Boosting"] = GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
            )
        
        if "Support Vector Machine" in models_to_train:
            C = st.sidebar.slider("SVM C:", 0.1, 10.0, 1.0)
            models_config["Support Vector Machine"] = SVR(C=C, kernel='rbf')
        
        if "K-Nearest Neighbors" in models_to_train:
            n_neighbors = st.sidebar.slider("KNN n_neighbors:", 1, 20, 5)
            models_config["K-Nearest Neighbors"] = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    # Check if any models are selected
    if not models_config:
        st.warning("Please select at least one model to train.")
        return {}
    
    # Train models
    for i, (model_name, model) in enumerate(models_config.items()):
        # Update status text if showing details
        if status_text is not None:
            status_text.text(f"Training {model_name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        
        # Store the trained model and its cross-validation scores
        trained_models[model_name] = {
            'model': model,
            'cv_scores': cv_scores,
            'training_time': training_time
        }
        
        # Update progress if showing details
        if progress_bar is not None:
            progress_value = (i + 1) / len(models_config)
            progress_bar.progress(progress_value)
        
        # Display training status if showing details
        if show_details:
            st.write(f"Trained {model_name} - Cross-validation RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Complete the progress indicators if showing details
    if show_details:
        if status_text is not None:
            status_text.text("Training completed!")
        if progress_bar is not None:
            progress_bar.empty()
    
    return trained_models
