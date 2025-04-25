# 🚗 Optifyx Car Price Predictor

**Optifyx Car Price Predictor** is an interactive web application built using **Streamlit** that predicts the resale price of a used car based on key input features. This project leverages multiple regression-based machine learning models to help users estimate a fair market price.

---

## 📈 Models Used

The following regression models are available in the app:

1. **Linear Regression**  
   A basic linear approach that models the relationship between features and price.

2. **Decision Tree Regressor**  
   A model that makes price predictions by learning decision rules from the data.

3. **Random Forest Regressor**  
   An ensemble learning model that builds multiple decision trees and averages the result.

4. **K-Nearest Neighbors (KNN)**  
   Predicts price based on the average price of nearest similar cars.

---

## 📊 Dataset

The model is trained on a cleaned dataset of used car listings. The dataset typically includes:

- **Year**: Year of manufacture
- **Present Price**: Showroom price of the car
- **Kms Driven**: Distance driven in kilometers
- **Fuel Type**: Petrol / Diesel / CNG
- **Seller Type**: Dealer or Individual
- **Transmission**: Manual or Automatic
- **Owner**: Number of previous owners
- **Selling Price**: Target variable

---

## ⚙️ Features

- Interactive Streamlit interface
- Dropdowns, sliders, and number inputs for easy data entry
- Multiple regression models to choose from
- Instant prediction of car resale price
- Display of model accuracy scores

---

## 🖥️ How to Run the App

1. Clone this repository:
   ```bash
   git clone https://github.com/Ansh07017/Optifyx-Car_Price_Predictor.git
   cd Optifyx-Car_Price_Predictor
2. pip install -r requirements.txt
3. streamlit run app.py

