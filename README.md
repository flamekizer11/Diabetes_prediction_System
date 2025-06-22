# Diabetes_prediction_System

# 🩺 Diabetes Prediction Web App
A modern Flask-based web application that predicts diabetes risk using machine learning and provides interactive data visualization comparing patient data with training data distributions.

## 📸 Screenshots
Here are some screenshots of the working application:

![Home Page](images/1.png)
*Main interface for entering patient data*

![Prediction Results](images/2.png)
*Diabetes prediction results with confidence score*

![Feature Analysis](images/3.png)
*Interactive data visualization showing patient data vs training data(feature analysis)*

## ✨ Features
- **Machine Learning Prediction**: Uses a trained model to predict diabetes risk
- **Interactive Web Interface**: Clean, responsive UI with gradient design
- **Data Visualization**: Scatter plots showing patient data vs training data across all features
- **Real-time Analysis**: Instant predictions with confidence scores
- **Feature Comparison**: Visual analysis of how patient data compares to diabetic/non-diabetic populations

## 🛠️ Technologies Used
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: scikit-learn, pickle
- **Data Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

## 📋 Prerequisites
- Python 3.7+
- pip package manager

## 🚀 Installation

### Clone the repository
```bash
git clone https://github.com/flamekizer11/Diabetes_prediction_System.git
cd Diabetes_prediciton_System
```

### Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies
```bash
pip install - requirements.txt
```

### Prepare your model
- Place your trained model file as `model.pkl` in the project root
- Optionally, add your training data as `diabetes_data.csv`

## 🏃‍♂️ Running the Application

### Start the Flask server
```bash
python app.py
```

### Open your browser
Navigate to `http://localhost:5000`

The application will be running locally!

## 📁 Project Structure
```
diabetes-prediction-flask/
│
├── app.py                # Main Flask application
├── model.pkl             # Trained ML model 
├── diabetes_data.csv     # Training data 
├── images/               # Screenshots of the application
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
├── templates/
│   └── index.html        # HTML template
├── static/
│   └── style.css         # CSS styles
└── README.md            # Project documentation
```

## 🎯 Usage

### Enter Patient Data
Input the 8 required features:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Get Prediction
Click "Predict Diabetes" to get:
- Diabetes risk prediction
- Confidence score

### View Analysis
Click "Show Feature Analysis" to see:
- Scatter plots comparing patient data with training data
- Visual representation across all 8 features

## 🎨 Features Explanation

### Input Features
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: Diabetes pedigree function score
- **Age**: Age in years

### Visualization
- **Blue dots**: Non-diabetic patients from training data
- **Coral dots**: Diabetic patients from training data
- **Red star**: Current patient data point

## 🔧 Customization

### Adding Your Own Model
Replace `model.pkl` with your trained model. Ensure it expects 8 features in the same order.

### Using Your Training Data
Replace the dummy data generation in `app.py` with:

```python
training_data = pd.read_csv('your_training_data.csv')
```

### Styling
Modify `static/style.css` to customize the appearance.

## 📊 Model Requirements
Your model should:
- Accept 8 numerical features
- Return binary predictions (0/1)
- Have a `predict_proba()` method for confidence scores
- Be saved using pickle

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer
This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.

## 🙏 Acknowledgments
- Machine learning model inspired by the Pima Indians Diabetes Database
- UI design inspired by modern web application trends
- Built with Flask framework and modern web technologies

## 📞 Support
If you encounter any issues or have questions:
- Check the Issues section
- Create a new issue if your problem isn't already reported
- Provide detailed information about the error and your environment

---
**Made with ❤️ for healthcare and machine learning enthusiasts**