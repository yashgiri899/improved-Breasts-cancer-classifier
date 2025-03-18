🎯 Advanced Breast Cancer Prediction using Random Forest Classifier

📌 Overview

This project builds an enhanced Random Forest Classifier to predict whether a tumor is malignant or benign using a medical dataset. It is an improved version of a previous implementation, featuring better preprocessing, hyperparameter tuning, and detailed model evaluation.

🚀 Features

Improved Data Preprocessing: Standardization with StandardScaler ensures better model performance.

Feature Selection & Engineering: Optimized feature selection for better predictions.

Hyperparameter Optimization: Fine-tuned n_estimators, max_depth, and min_samples_split for better accuracy.

Evaluation Metrics: Accuracy, Confusion Matrix, Precision, Recall, and F1-score are used to assess the model.

Cross-Validation: Ensures the model's reliability across different data splits.

Feature Importance Analysis: Identifies the most influential factors in cancer prediction.

Better Code Structure: Clean, modular, and reusable code.

👤 Project Structure

📺 BreastCancer_Classification/
│── 📄 cancer.py                # Main script for training & evaluation  
│── 📄 Data (2).csv              # Breast cancer dataset  
│── 📄 README.md               # Documentation (You are here!)  

📊 Dataset Information

The dataset contains features extracted from cell nuclei in breast cancer biopsy samples. Key attributes include:

Clump Thickness

Uniformity of Cell Size

Uniformity of Cell Shape

Marginal Adhesion

Bland Chromatin

Normal Nucleoli

Mitoses

Target Variable: Benign (0) / Malignant (1)

⚙️ Installation & Setup

🔹 Step 1: Clone the Repository

git clone https://github.com/yashgiri899/BreastCancer_Classification.git  
cd BreastCancer_Classification  

🔹 Step 2: Run the Classifier

python cancer.py  

📊 Model Performance

✅ Accuracy: 97.8% (Higher than previous implementations!)

✅ Confusion Matrix:

[[84  3]
 [ 0 50]]

✅ Precision, Recall, and F1-Score available in output logs.

📌 Improvements Over Previous Version

✅ Optimized Random Forest Hyperparameters (Better n_estimators, random_state)✅ More Robust Preprocessing (StandardScaler applied correctly)✅ Enhanced Model Evaluation (More metrics analyzed)✅ Feature Importance Analysis Included✅ More Modular Codebase for Future Enhancements

🐝 License

This project is licensed under the MIT License.

👨‍💻 Author

YASH VARDHAN GIRI

📧 yashgiri803@gmail.com

🌐 LinkedIn: https://www.linkedin.com/in/yash-giri-987072326/

🌟 Feel free to contribute and star this repository if you found it helpful! 🌟

