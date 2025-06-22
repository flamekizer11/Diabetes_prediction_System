from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Load training data (you'll need to save this during training)
try:
    training_data = pd.read_csv('diabetes_data.csv')  # Replace with your training data file
except:
    # If no training data file, create dummy data for demonstration
    np.random.seed(42)
    training_data = pd.DataFrame({
        'pregnancies': np.random.randint(0, 18, 768),
        'glucose': np.random.normal(120, 32, 768),
        'bloodpressure': np.random.normal(72, 12, 768),
        'skinthickness': np.random.normal(23, 16, 768),
        'insulin': np.random.normal(80, 115, 768),
        'bmi': np.random.normal(32, 7, 768),
        'dpf': np.random.normal(0.47, 0.33, 768),
        'age': np.random.randint(21, 82, 768),
        'outcome': np.random.choice([0, 1], 768)
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]
        result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
        confidence = model.predict_proba([features])[0][prediction] * 100
        
        # Store patient data in session for plotting
        patient_data = {
            'pregnancies': features[0],
            'glucose': features[1], 
            'bloodpressure': features[2],
            'skinthickness': features[3],
            'insulin': features[4],
            'bmi': features[5],
            'dpf': features[6],
            'age': features[7]
        }
        
        return render_template('index.html', 
                             prediction_text=f'Patient is: {result}',
                             confidence=f'Confidence: {confidence:.1f}%',
                             patient_data=patient_data,
                             show_plot_btn=True)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

@app.route('/generate_plots', methods=['POST'])
def generate_plots():
    try:
        # Get patient data from form
        patient_data = {
            'pregnancies': float(request.form['pregnancies']),
            'glucose': float(request.form['glucose']),
            'bloodpressure': float(request.form['bloodpressure']),
            'skinthickness': float(request.form['skinthickness']),
            'insulin': float(request.form['insulin']),
            'bmi': float(request.form['bmi']),
            'dpf': float(request.form['dpf']),
            'age': float(request.form['age'])
        }
        
        # Generate plots
        plots = []
        feature_names = list(patient_data.keys())
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Patient Data vs Training Data Distribution', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(feature_names):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            # Plot training data as scatter plot
            diabetic_data = training_data[training_data['outcome'] == 1]
            non_diabetic_data = training_data[training_data['outcome'] == 0]
            
            # Create y-values for scatter (random jitter)
            np.random.seed(42)
            diabetic_y = np.random.normal(1, 0.1, len(diabetic_data))
            non_diabetic_y = np.random.normal(0, 0.1, len(non_diabetic_data))
            
            scatter1 = ax.scatter(non_diabetic_data[feature], non_diabetic_y, alpha=0.6, 
                      color='lightblue', s=20)
            scatter2 = ax.scatter(diabetic_data[feature], diabetic_y, alpha=0.6, 
                      color='lightcoral', s=20)
            
            # Plot patient data point
            scatter3 = ax.scatter(patient_data[feature], 0.5, color='red', s=200, 
                      marker='*', edgecolors='darkred', linewidth=2)
            
            ax.set_title(feature.title())
            ax.set_xlabel(feature.title())
            ax.set_ylabel('Outcome')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Non-Diabetic', 'Diabetic'])
            ax.grid(True, alpha=0.3)
        
        # Add single legend outside plots
        fig.legend([scatter1, scatter2, scatter3], 
                  ['Non-Diabetic', 'Diabetic', 'Patient'], 
                  loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({'plot': plot_url})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)