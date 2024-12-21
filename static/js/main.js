document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners to species checkboxes
    document.querySelectorAll('.species-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const settingsDiv = this.closest('.species-container').querySelector('.species-settings');
            settingsDiv.style.display = this.checked ? 'block' : 'none';
        });
    });

    // Add event listener to generate button
    document.getElementById('generateBtn').addEventListener('click', async function() {
        await generateData();
        await generatePlots(); // Call to generate plots after data generation
    });

    // Add plot button listeners
    document.querySelectorAll('[data-plot]').forEach(button => {
        button.addEventListener('click', (e) => {
            updatePlotDisplay(e.target.dataset.plot);
        });
    });

    // Add these event listeners to the DOMContentLoaded function
    document.getElementById('modelUploadForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        uploadModel(formData);
    });

    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        try {
            const formData = new FormData(this);
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }
            
            const resultDiv = document.getElementById('predictionResult');
            let probabilitiesHtml = '<h5>Probabilities:</h5><ul>';
            
            // Sort probabilities in descending order
            const sortedProbabilities = Object.entries(data.probabilities)
                .sort((a, b) => b[1] - a[1]);

            for (const [species, probability] of sortedProbabilities) {
                const percentage = (probability * 100).toFixed(2);
                probabilitiesHtml += `
                    <li>${species}: ${percentage}%
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: ${percentage}%" 
                                 aria-valuenow="${percentage}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="100">
                            </div>
                        </div>
                    </li>`;
            }
            probabilitiesHtml += '</ul>';

            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <h4>Predicted Species: ${data.prediction}</h4>
                    ${probabilitiesHtml}
                </div>
            `;
            resultDiv.style.display = 'block';
            
        } catch (error) {
            showError(error.message);
        }
    });

    document.getElementById('submitParameters').addEventListener('click', function() {
        const meanLength = parseFloat(document.getElementById('meanLength').value);
        const meanWidth = parseFloat(document.getElementById('meanWidth').value);
        const colorIntensity = parseFloat(document.getElementById('colorIntensity').value);
        const meanFinLength = parseFloat(document.getElementById('meanFinLength').value);

        // Create an object to send to the server
        const parameters = {
            meanLength,
            meanWidth,
            colorIntensity,
            meanFinLength
        };

        // Send the parameters to the server
        fetch('/submit_fish_parameters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(parameters)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            // Handle success (e.g., show a message or update the UI)
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });
});

async function generateData() {
    document.getElementById('spinner').style.display = 'block'; // Show spinner
    const selectedSpecies = {};
    let speciesCount = 0;

    document.querySelectorAll('.species-checkbox:checked').forEach(checkbox => {
        const species = checkbox.value;
        const container = checkbox.closest('.species-container');
        
        selectedSpecies[species] = {
            enabled: true,
            length_mean: parseFloat(container.querySelector(`#${species}_length`).value),
            width_mean: parseFloat(container.querySelector(`#${species}_width`).value),
            color_intensity: parseFloat(container.querySelector(`#${species}_color`).value),
            fin_length: parseFloat(container.querySelector(`#${species}_fin`).value)
        };
        speciesCount++;
    });

    if (speciesCount < 2) {
        showError('Please select at least 2 fish species');
        return;
    }

    const params = {
        species_settings: selectedSpecies,
        n_samples: parseInt(document.getElementById('n_samples').value),
        test_size: parseInt(document.getElementById('test_size').value) / 100
    };

    try {
        document.getElementById('generateBtn').disabled = true;
        
        const response = await fetch('/generate_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to generate data');
        }

        displayResults(data);
        hideError();

    } catch (error) {
        showError(error.message);
        console.error('Error:', error);
    } finally {
        document.getElementById('generateBtn').disabled = false;
        document.getElementById('spinner').style.display = 'none'; // Hide spinner
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    document.getElementById('errorAlert').style.display = 'none';
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';
    
    // Update dataset information
    document.getElementById('datasetInfo').innerHTML = `
        <div class="col-md-4">
            <h4>Total Samples</h4>
            <p>${data.dataset_info.total_samples}</p>
        </div>
        <div class="col-md-4">
            <h4>Training Samples</h4>
            <p>${data.dataset_info.training_samples} (${Math.round((data.dataset_info.training_samples / data.dataset_info.total_samples) * 100)}%)</p>
        </div>
        <div class="col-md-4">
            <h4>Testing Samples</h4>
            <p>${data.dataset_info.testing_samples} (${Math.round((data.dataset_info.testing_samples / data.dataset_info.total_samples) * 100)}%)</p>
        </div>
    `;

    // Display generated data sample
    const dataSampleDiv = document.getElementById('dataSample');
    dataSampleDiv.innerHTML = `
        <h5>Sample Data:</h5>
        <div class="sample-data-container">
            <pre class="sample-data-content">${JSON.stringify(data.data_sample, null, 2)}</pre>
        </div>
    `;

    // Update model results with all metrics
    const modelResults = document.getElementById('modelResults');
    modelResults.innerHTML = '';
    
    Object.entries(data.results).forEach(([model, result]) => {
        const accuracy = (result.accuracy * 100).toFixed(2);
        modelResults.innerHTML += `
            <div class="model-section mb-4">
                <h4>${model}</h4>
                <div class="progress mb-3">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${accuracy}%" 
                         aria-valuenow="${accuracy}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                        ${accuracy}%
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <h5>Confusion Matrix</h5>
                        <img src="${result.confusion_matrix}" class="img-fluid" alt="Confusion Matrix">
                    </div>
                    <div class="col-md-6">
                        <h5>Learning Curve</h5>
                        <img src="${result.learning_curve}" class="img-fluid" alt="Learning Curve">
                    </div>
                </div>

                <div class="mt-3">
                    <button class="btn btn-secondary" onclick="downloadFile('${model}_model')">
                        Download ${model} Model
                    </button>
                </div>
            </div>
        `;
    });

    // Highlight best model
    if (data.best_model) {
        const bestModelDiv = document.createElement('div');
        bestModelDiv.className = 'alert alert-info';
        bestModelDiv.innerHTML = `<h4>Best Model: ${data.best_model}</h4>`;
        modelResults.prepend(bestModelDiv);
    }

    // Update feature visualization
    if (data.plots) {
        updatePlotDisplay('2d'); // Default to 2D plot
    }

    // Update download section
    document.getElementById('downloadSection').innerHTML = `
        <div class="row">
            <div class="col-md-4">
                <button class="btn btn-secondary btn-block mb-2" onclick="downloadFile('original_dataset')">
                    Download Original Dataset
                </button>
            </div>
            <div class="col-md-4">
                <button class="btn btn-secondary btn-block mb-2" onclick="downloadFile('scaled_dataset')">
                    Download Scaled Dataset
                </button>
            </div>
            <div class="col-md-4">
                <button class="btn btn-secondary btn-block mb-2" onclick="downloadFile('scaler')">
                    Download Scaler
                </button>
            </div>
        </div>
    `;

    // Update performance metrics summary
    const performanceSummaryDiv = document.getElementById('performanceSummary');
    performanceSummaryDiv.innerHTML = '<h5>Metrics:</h5><ul>';
    Object.entries(data.results).forEach(([model, result]) => {
        performanceSummaryDiv.innerHTML += `<li>${model}: ${result.accuracy.toFixed(2)}% accuracy</li>`;
    });
    performanceSummaryDiv.innerHTML += '</ul>';
}

function downloadFile(fileType) {
    window.location.href = `/download/${fileType}`;
}

// Function to generate plots
async function generatePlots() {
    const response = await fetch('/generate_plots', {
        method: 'POST',
    });

    if (!response.ok) {
        throw new Error('Failed to generate plots');
    }
}

// Function to update plot display
function updatePlotDisplay(plotType) {
    const plotArea = document.getElementById('plotArea');
    const plot2dBtn = document.querySelector('[data-plot="2d"]');
    const plot3dBtn = document.querySelector('[data-plot="3d"]');

    // Update the plot display
    if (plotType === '2d') {
        plotArea.innerHTML = `<iframe src="/static/plots/2d_plot.html" 
                                    frameborder="0" 
                                    style="width: 100%; height: 600px;">
                             </iframe>`;
        plot2dBtn.classList.add('active');
        plot3dBtn.classList.remove('active');
    } else {
        plotArea.innerHTML = `<iframe src="/static/plots/3d_plot.html" 
                                    frameborder="0" 
                                    style="width: 100%; height: 600px;">
                             </iframe>`;
        plot3dBtn.classList.add('active');
        plot2dBtn.classList.remove('active');
    }
}

async function uploadModel(formData) {
    try {
        const response = await fetch('/upload_model', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        showSuccess('Model and scaler uploaded successfully');
        document.getElementById('predictionForm').style.display = 'block';
        
    } catch (error) {
        showError(error.message);
    }
}

function showSuccess(message) {
    const successDiv = document.getElementById('successAlert');
    const successMessage = document.getElementById('successMessage');
    successMessage.textContent = message;
    successDiv.style.display = 'block';
    setTimeout(() => {
        successDiv.style.display = 'none';
    }, 5000);
}

$(document).ready(function() {
    $('.btn-group .btn').click(function() {
        var plotType = $(this).data('plot');
        if (plotType === '2d') {
            $('#2dPlot').show();
            $('#3dPlot').hide();
        } else {
            $('#2dPlot').hide();
            $('#3dPlot').show();
        }
    });
});