# Python Libraries
from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from werkzeug.utils import secure_filename
from mpl_toolkits.mplot3d import Axes3D
import pickle
import logging

logging.basicConfig(level=logging.ERROR)

app = Flask(__name__, static_url_path='/static')

# Necessary directories
for directory in ['models', 'uploads', 'static/plots', 'static/metrics']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Fish species and their default characteristics
FISH_SPECIES = {
    'Betta': {
        'length_mean': 0.0,
        'width_mean': 0.0,
        'color_intensity': 0.0,
        'fin_length': 0.0,
        'std_dev': 0.3
    },
    'Arowana': {
        'length_mean': 0.0,
        'width_mean': 0.0,
        'color_intensity': 0.0,
        'fin_length': 0.0,
        'std_dev': 0.3
    },
    'Goldfish': {
        'length_mean': 0.0,
        'width_mean': 0.0,
        'color_intensity': 0.0,
        'fin_length': 0.0,
        'std_dev': 0.3
    },
    'Koi': {
        'length_mean': 0.0,
        'width_mean': 0.0,
        'color_intensity': 0.0,
        'fin_length': 0.0,
        'std_dev': 0.3
    }
}

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pkl'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html', fish_species=FISH_SPECIES)

@app.route('/generate_data', methods=['POST'])
def generate_data():
    try:
        data = request.get_json()
        species_settings = data.get('species_settings', {})
        selected_species = list(species_settings.keys())  # Get selected species
        n_samples = int(data.get('n_samples', 1000))
        test_size = float(data.get('test_size', 0.3))

        if len(species_settings) < 2:
            return jsonify({'error': 'Please select at least 2 fish species'}), 400

        # Calculate the number of samples per species
        n_species_samples = [n_samples // len(species_settings)] * len(species_settings)
        for i in range(n_samples % len(species_settings)):
            n_species_samples[i] += 1

        # Generate synthetic data
        X = []
        y = []
        sample_data = []

        for i, (species, settings) in enumerate(species_settings.items()):
            length_mean = float(settings.get('length_mean'))
            width_mean = float(settings.get('width_mean'))
            color_intensity = float(settings.get('color_intensity'))
            fin_length = float(settings.get('fin_length'))
            std_dev = float(settings.get('std_dev', 0.1))

            # Generate species data with normal distribution
            species_data = np.random.normal(
                loc=[length_mean, width_mean, color_intensity, fin_length],
                scale=[length_mean * std_dev, width_mean * std_dev, 10.0, fin_length * std_dev],
                size=(n_species_samples[i], 4)
            )

            # Ensure no negative values
            species_data = np.clip(species_data, a_min=0, a_max=None)

            X.extend(species_data)
            y.extend([species] * n_species_samples[i])

            # Append labeled data to sample_data
            for j in range(n_species_samples[i]):
                sample_data.append({
                    'Species': species,
                    'Color Intensity': species_data[j][2],
                    'Fin Length': species_data[j][3],
                    'Length': species_data[j][0],
                    'Width': species_data[j][1]
                })

        # Save selected species separately
        with open('models/selected_species.pkl', 'wb') as f:
            pickle.dump(selected_species, f)

        # Convert y to a numpy array
        y = np.array(y)

        # Ensure only selected species are used
        unique_species = np.unique(y)
        if not all(species in selected_species for species in unique_species):
            return jsonify({'error': 'Unexpected species in data'}), 400

        X = np.array(X)

        # Add noise to make data more realistic
        noise_factor = 0.05
        X = X + np.random.normal(0, noise_factor, X.shape)

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models and generate metrics
        models = {
            'GradientBoosting': GradientBoostingClassifier(),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Add depth limit
                min_samples_split=5,  # Minimum samples required to split
                min_samples_leaf=2,  # Minimum samples required at leaf node
                random_state=42  # For consistency
            ),
            'SVM': SVC(probability=True)
        }

        model_results = {}
        best_model = None
        best_accuracy = 0

        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = np.mean(y_pred == y_test)

            # Save model if it's the best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name
                
            # Always save each model with selected species information
            model_info = {
                'model': model,
                'selected_species': selected_species,
                'classes_': model.classes_
            }
            joblib.dump(model_info, f'models/{name}_model.pkl')

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(y),
                        yticklabels=np.unique(y))
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'static/metrics/{name}_confusion_matrix.png')
            plt.close()

            # Learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train_scaled, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5)
            )
            plt.figure()
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
            plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation Score')
            plt.title(f'Learning Curves - {name}')
            plt.xlabel('Training Examples')
            plt.ylabel('Score')
            plt.legend(loc='best')
            plt.savefig(f'static/metrics/{name}_learning_curve.png')
            plt.close()

            model_results[name] = {
                'accuracy': accuracy,
                'confusion_matrix': f'/static/metrics/{name}_confusion_matrix.png',
                'learning_curve': f'/static/metrics/{name}_learning_curve.png'
            }

        # Save datasets and scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        pd.DataFrame(X, columns=['Length', 'Width', 'Color Intensity', 'Fin Length']).to_csv('uploads/original_dataset.csv', index=False)
        pd.DataFrame(scaler.transform(X), columns=['Length', 'Width', 'Color Intensity', 'Fin Length']).to_csv('uploads/scaled_dataset.csv', index=False)

        # Generate plots
        generate_2d_plot(X, y)
        generate_3d_plot(X, y)

        # Return results with labeled sample data
        return jsonify({
            'success': True,
            'results': model_results,
            'best_model': best_model,
            'dataset_info': {
                'total_samples': len(X),
                'training_samples': len(X_train),
                'testing_samples': len(X_test)
            },
            'data_sample': sample_data,  # Return labeled sample data
            'plots': {
                '2d': '/static/plots/2d_plot.html',
                '3d': '/static/plots/3d_plot.html'
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<file_type>')
def download_file(file_type):
    try:
        if file_type == 'original_dataset':
            return send_file('uploads/original_dataset.csv', 
                           mimetype='text/csv',
                           as_attachment=True,
                           download_name='original_dataset.csv')
        elif file_type == 'scaled_dataset':
            return send_file('uploads/scaled_dataset.csv',
                           mimetype='text/csv',
                           as_attachment=True,
                           download_name='scaled_dataset.csv')
        elif file_type.endswith('_model'):
            model_name = file_type.replace('_model', '')
            return send_file(f'models/{model_name}_model.pkl',
                           as_attachment=True,
                           download_name=f'{model_name}_model.pkl')
        elif file_type == 'scaler':
            return send_file('models/scaler.pkl',
                           as_attachment=True,
                           download_name='scaler.pkl')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'model' not in request.files or 'scaler' not in request.files:
            return jsonify({'error': 'Both model and scaler files are required'}), 400

        model_file = request.files['model']
        scaler_file = request.files['scaler']

        if model_file.filename == '' or scaler_file.filename == '':
            return jsonify({'error': 'No selected files'}), 400

        if model_file and allowed_file(model_file.filename) and scaler_file and allowed_file(scaler_file.filename):
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_model.pkl')
            scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_scaler.pkl')
            
            # Save the files
            model_file.save(model_path)
            scaler_file.save(scaler_path)

            # Verify model contains selected species information
            try:
                model_info = joblib.load(model_path)
                if not isinstance(model_info, dict) or 'selected_species' not in model_info:
                    return jsonify({'error': 'Invalid model format or missing species information'}), 400
            except Exception as e:
                return jsonify({'error': 'Failed to load model information'}), 400

            return jsonify({'success': True, 'message': 'Files uploaded successfully'})

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form if request.form else request.get_json()
        
        features = np.array([[
            float(data.get('length', 0)),
            float(data.get('width', 0)),
            float(data.get('color_intensity', 0)),
            float(data.get('fin_length', 0))
        ]])

        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_model.pkl')
        scaler_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_scaler.pkl')

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            return jsonify({'error': 'Model and scaler must be uploaded first'}), 400

        # Load model info
        model_info = joblib.load(model_path)
        model = model_info['model']
        selected_species = model_info['selected_species']
        
        # Ensure model classes match selected species
        if not all(species in model.classes_ for species in selected_species):
            return jsonify({'error': 'Model classes do not match selected species'}), 400
        
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(features)
        
        # Get raw probabilities for all classes
        raw_probabilities = model.predict_proba(features_scaled)[0]
        
        # Create probability dictionary for all classes
        prob_dict = {species: float(prob) for species, prob in zip(model.classes_, raw_probabilities)}
        
        # Get prediction
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': prob_dict
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def generate_2d_plot(X, y):
    df = pd.DataFrame(X, columns=['Length', 'Width', 'Color Intensity', 'Fin Length'])
    df['Species'] = y
    
    # 2D scatter plot
    fig = px.scatter(df, 
                    x='Length', 
                    y='Width',
                    color='Species',
                    title='Fish Species Distribution (2D View)',
                    labels={'Length': 'Length (cm)',
                           'Width': 'Width (cm)'},
                    hover_data=['Color Intensity', 'Fin Length'],
                    template='plotly_dark')
    
    # This part is the layout for better looks
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Save as HTML file
    fig.write_html('static/plots/2d_plot.html')

def generate_3d_plot(X, y):
    # DataFrame for  plotting
    df = pd.DataFrame(X, columns=['Length', 'Width', 'Color Intensity', 'Fin Length'])
    df['Species'] = y
    
    # 3D scatter plot
    fig = px.scatter_3d(df,
                       x='Length',
                       y='Width',
                       z='Color Intensity',
                       color='Species',
                       title='Fish Species Distribution (3D View)',
                       labels={'Length': 'Length (cm)',
                              'Width': 'Width (cm)',
                              'Color Intensity': 'Color Intensity'},
                       hover_data=['Fin Length'],
                       template='plotly_dark')
    
    # Update layout for better appearance
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)',
                      gridcolor='white',
                      showbackground=True),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)',
                      gridcolor='white',
                      showbackground=True),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)',
                      gridcolor='white',
                      showbackground=True)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Save as HTML file for interactivity
    fig.write_html('static/plots/3d_plot.html')

@app.route('/submit_fish_parameters', methods=['POST'])
def submit_fish_parameters():
    data = request.get_json()
    mean_length = data.get('meanLength')
    mean_width = data.get('meanWidth')
    color_intensity = data.get('colorIntensity')
    mean_fin_length = data.get('meanFinLength')

    # Process the parameters as needed
    # For example, you can save them or use them for predictions

    return jsonify({'status': 'success', 'message': 'Parameters received successfully'})

@app.route('/generate_plots', methods=['POST'])
def generate_plots():
    # Example data for plotting
    X = np.random.rand(100, 2)  # Replace with your actual data
    y = np.random.randint(0, 2, 100)  # Replace with your actual labels

    # Generate 2D plot
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title('2D Plot')
    plt.savefig('static/plots/2d_plot.png')
    plt.close()

    # Generate 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], np.random.rand(100), c=y)
    ax.set_title('3D Plot')
    plt.savefig('static/plots/3d_plot.png')
    plt.close()

    return jsonify({'success': True})

@app.route('/plots/<plot_type>')
def get_plot(plot_type):
    if plot_type == '2d':
        return send_from_directory('static/plots', '2d_plot.png')
    elif plot_type == '3d':
        return send_from_directory('static/plots', '3d_plot.png')
    return jsonify({'error': 'Plot not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)