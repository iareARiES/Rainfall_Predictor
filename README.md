# ☂️ Rainfall Prediction Using Machine Learning 🌧️

## 📖 Important Note
For a **better understanding** of the project, please check the **Google Colab file** 📄 uploaded in this repository. It contains **detailed explanations and execution steps** to help you grasp the workflow more effectively.

This repository contains a machine learning model that predicts rainfall based on various weather parameters. The model is built using a Random Forest Classifier and is fine-tuned using GridSearchCV to optimize performance. 👩‍💻🌍

## 🌟 Features 
- 📊 Data preprocessing, including handling missing values and encoding categorical data
- 📊 Exploratory Data Analysis (EDA) with visualization
- 🎨 Model training and hyperparameter tuning
- ⚡️ Downsampling to handle class imbalance
- 🔢 Model evaluation using accuracy, confusion matrix, and classification report
- 📊 Prediction on new data inputs
- 📂 Model serialization using `pickle` for later use

## 🛠️ Requirements
Make sure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 💪 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rainfall-prediction.git
   cd rainfall-prediction
   ```
2. Run the model training script:
   ```bash
   python rainfall_prediction.py
   ```
3. Load and use the saved model:
   ```python
   import pickle
   import pandas as pd
   
   with open("rainfall_prediction_model.pkl", "rb") as file:
       model_data = pickle.load(file)
   
   model = model_data["model"]
   feature_names = model_data["feature_names"]
   
   input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
   input_df = pd.DataFrame([input_data], columns=feature_names)
   prediction = model.predict(input_df)
   print("Prediction result:", "☔️ Rainfall" if prediction[0] == 1 else "🌞 No Rainfall")
   ```


## 🏆 License
This project is open-source and available under the [MIT License](LICENSE). 💎

