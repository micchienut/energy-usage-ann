# Office Energy Usage Prediction ⚡  
This project was developed as part of my Deep Learning mid exam. The goal is to predict office energy consumption using an Artificial Neural Network (ANN) model.  

## ✨ Project Overview  
- **Dataset**: 1,229 rows × 12 columns (features include Month, Hour, DayOfWeek, Holiday, Temperature, Humidity, Square Footage, Occupancy, HVACUsage, LightingUsage, Renewable Energy, EnergyConsumption).  
- **Target Variable**: EnergyConsumption  
- **Models**:  
  - Sequential Baseline Model (3 hidden layers) → ~20–25% R²  
  - Sequential Tuned Model (hyperparameter tuning with KerasTuner) → improved convergence and reduced overfitting  
  - Functional Baseline & Modified Model → struggled compared to Sequential models  
- **Key Insight**: Overcomplicated architectures on small datasets (≈1200 rows) may lead to poor generalization.  

## 📊 Key Findings  
- Sequential models consistently performed better than functional ones.  
- Tuning (neurons, dropout, learning rate) improved results and reduced overfitting.  
- Matching scaling method and activation function matters:  
  - MinMaxScaler on target → worked best with Sigmoid output.  
- Functional modified model performed poorly (low R²), likely due to too many neurons and mismatched activation function.  
- R² scores across best models: ~20–25% variance explained.  

⚠️ **Challenge**: Limited dataset size made it difficult to build a highly accurate regression model.  

## 🛠️ Tech Stack  
- Python  
- TensorFlow & Keras  
- Keras Tuner  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib & Seaborn  

## 🚀 Future Improvements  
- Collect more data (1200 rows too small for complex ANN).  
- Experiment with simpler models (e.g., Linear Regression, Random Forest) as baselines.  
- Adjust architectures carefully to avoid overfitting with small datasets.  
- Test other activation functions like LeakyReLU in a more controlled setup.

## 🎥 Video Explanation
Video explanation can be seen [here](https://drive.google.com/file/d/1pHm3FKKOcD4eauC8WoVlC7H8Bo1_liZN/view?usp=drive_link)

## 👩‍💻 Author  
**Michelle Nathania**  
Data Science student at BINUS University | Interested in Machine Learning and Deep Learning

---
