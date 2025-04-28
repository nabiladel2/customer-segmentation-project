/**
 * Customer Segmentation Dashboard JavaScript
 * 
 * This script handles the frontend functionality of the customer segmentation
 * application, including data upload, preprocessing, clustering, and visualization.
 */

// API endpoint (adjust based on your deployment)
const API_BASE_URL = '';  // Empty for same-origin, or set to your API URL

// Global variables to track application state
let currentStep = 1;
let features = [];
let selectedFeatures = [];
let optimalK = null;

// Document ready function
$(document).ready(function() {
    // Bind event handlers
    bindEventHandlers();
    
    // Initialize the UI
    updateStepIndicators();
});

/**
 * Binds all event handlers for the application
 */
function bindEventHandlers() {
    // Upload button click
    $('#upload-btn').click(handleDataUpload);
    
    // Algorithm selection change
    $('#algorithm-select').change(handleAlgorithmChange);
    
    // Preprocess button click
    $('#preprocess-btn').click(handlePreprocessing);
    
    // Find optimal K button click
    $('#find-optimal-k').click(findOptimalK);
    
    // Cluster button click
    $('#cluster-btn').click(performClustering);
    
    // Export buttons click
    $('#export-btn').click(exportResults);
    $('#weka-btn').click(exportWeka);
    
    // Restart button click
    $('#restart-btn').click(restartApplication);
}

/**
 * Updates the step indicators based on current step
 */
function updateStepIndicators() {
    // Reset all steps
    $('.step').removeClass('active completed');
    
    // Mark completed steps
    for (let i = 1; i < currentStep; i++) {
        $(`#step-${i}`).addClass('completed');
    }
    
    // Mark current step
    $(`#step-${currentStep}`).addClass('active');
    
    // Show/hide sections based on current step
    $('.row[id$="-section"]').hide();
    switch (currentStep) {
        case 1:
            $('#upload-section').show();
            break;
        case 2:
            $('#preprocess-section').show();
            break;
        case 3:
            $('#clustering-section').show();
            break;
        case 4:
            $('#results-section').show();
            break;
    }
}

/**
 * Handles file upload and data preview
 */
function handleDataUpload() {
    const fileInput = $('#file-input')[0];
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('Please select a file to upload.');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading();
    
    // Send the file to the server
    $.ajax({
        url: `${API_BASE_URL}/upload`,
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            hideLoading();
            
            if (response.success) {
                // Display data preview
                populateDataPreview(response.data_info);
                
                // Update features list
                features = response.data_info.columns;
                populateFeatureSelector(features);
                
                // Move to next step
                currentStep = 2;
                updateStepIndicators();
            } else {
                showError(response.error || 'Error uploading file.');
            }
        },
        error: function(xhr) {
            hideLoading();
            showError('Error uploading file: ' + (xhr.responseJSON?.error || xhr.statusText));
        }
    });
}

/**
 * Populates the data preview table with the first few rows
 */
function populateDataPreview(dataInfo) {
    const table = $('#data-preview-table');
    
    // Clear existing table
    table.find('thead tr').empty().append('<th>#</th>');
    table.find('tbody').empty();
    
    // Add columns
    dataInfo.columns.forEach(column => {
        table.find('thead tr').append(`<th>${column}</th>`);
    });
    
    // Add rows
    dataInfo.preview.forEach((row, index) => {
        const rowHtml = $('<tr>').append(`<td>${index + 1}</td>`);
        
        dataInfo.columns.forEach(column => {
            rowHtml.append(`<td>${row[column] !== null ? row[column] : 'N/A'}</td>`);
        });
        
        table.find('tbody').append(rowHtml);
    });
}

/**
 * Populates the feature selector with checkboxes
 */
function populateFeatureSelector(features) {
    const selector = $('#feature-selector');
    selector.empty();
    
    features.forEach(feature => {
        const checkboxItem = `
            <div class="form-check">
                <input class="form-check-input feature-checkbox" type="checkbox" value="${feature}" id="feature-${feature}">
                <label class="form-check-label" for="feature-${feature}">
                    ${feature}
                </label>
            </div>
        `;
        selector.append(checkboxItem);
    });
    
    // Add event handler for checkboxes
    $('.feature-checkbox').change(function() {
        selectedFeatures = $('.feature-checkbox:checked').map(function() {
            return $(this).val();
        }).get();
    });
    
    // Select all numeric features by default (assuming they are numeric)
    $('.feature-checkbox').prop('checked', true).trigger('change');
}

/**
 * Handles data preprocessing
 */
function handlePreprocessing() {
    if (selectedFeatures.length === 0) {
        showError('Please select at least one feature for clustering.');
        return;
    }
    
    showLoading();
    
    // Send preprocessing request
    $.ajax({
        url: `${API_BASE_URL}/preprocess`,
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            features: selectedFeatures
        }),
        success: function(response) {
            hideLoading();
            
            if (response.success) {
                // Display data statistics
                displayDataStats(response.stats, response.features_used);
                
                // Update selected features (in case some were excluded)
                selectedFeatures = response.features_used;
                
                // Move to next step
                currentStep = 3;
                updateStepIndicators();
            } else {
                showError(response.error || 'Error preprocessing data.');
            }
        },
        error: function(xhr) {
            hideLoading();
            showError('Error preprocessing data: ' + (xhr.responseJSON?.error || xhr.statusText));
        }
    });
}

/**
 * Displays data statistics after preprocessing
 */
function displayDataStats(stats, featuresUsed) {
    const statsContainer = $('#data-stats');
    statsContainer.empty();
    
    // Add feature info
    statsContainer.append(`<p><strong>Features used:</strong> ${featuresUsed.length}</p>`);
    
    // Create table for feature statistics
    const tableHtml = `
        <div class="table-container">
            <table class="table table-sm table-bordered table-hover">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody id="stats-table-body">
                </tbody>
            </table>
        </div>
    `;
    statsContainer.append(tableHtml);
    
    // Add rows for each feature
    const tableBody = $('#stats-table-body');
    featuresUsed.forEach(feature => {
        const featureStats = stats.feature_stats[feature];
        const row = `
            <tr>
                <td>${feature}</td>
                <td>${featureStats.mean.toFixed(2)}</td>
                <td>${featureStats.std.toFixed(2)}</td>
                <td>${featureStats.min.toFixed(2)}</td>
                <td>${featureStats.max.toFixed(2)}</td>
            </tr>
        `;
        tableBody.append(row);
    });
}

/**
 * Handles algorithm selection change
 */
function handleAlgorithmChange() {
    const algorithm = $('#algorithm-select').val();
    
    // Hide all parameter divs
    $('[id$="-params"]').hide();
    
    // Show parameters for selected algorithm
    $(`#${algorithm}-params`).show();
}

/**
 * Finds the optimal K for K-means clustering
 */
function findOptimalK() {
    showLoading();
    
    // Send optimal K request
    $.ajax({
        url: `${API_BASE_URL}/optimal_k`,
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({}),
        success: function(response) {
            hideLoading();
            
            if (response.success) {
                // Display elbow plot
                const plotHtml = `
                    <div class="text-center mt-3 mb-3">
                        <h5>Elbow Method and Silhouette Score</h5>
                        <img src="data:image/png;base64,${response.result.plot}" alt="Elbow Method Plot" class="img-fluid">
                    </div>
                `;
                $('#data-stats').append(plotHtml);
                
                // Find elbow point (optimal K)
                const kValues = response.result.k_values;
                const inertiaValues = response.result.inertia_values;
                const silhouetteValues = response.result.silhouette_values;
                
                // Use silhouette method (maximum silhouette score)
                const optimalKIndex = silhouetteValues.indexOf(Math.max(...silhouetteValues));
                optimalK = kValues[optimalKIndex];
                
                // Update K input
                $('#kmeans-k').val(optimalK);
                
                // Show recommendation
                const recommendationHtml = `
                    <div class="alert alert-info mt-3">
                        <i class="fas fa-info-circle me-2"></i>
                        Recommended number of clusters (K): <strong>${optimalK}</strong> 
                        (based on maximum silhouette score of ${silhouetteValues[optimalKIndex].toFixed(3)})
                    </div>
                `;
                $('#data-stats').append(recommendationHtml);
            } else {
                showError(response.error || 'Error finding optimal K.');
            }
        },
        error: function(xhr) {
            hideLoading();
            showError('Error finding optimal K: ' + (xhr.responseJSON?.error || xhr.statusText));
        }
    });
}

/**
 * Performs clustering based on selected algorithm and parameters
 */
function performClustering() {
    const algorithm = $('#algorithm-select').val();
    let params = {};
    
    // Get parameters based on algorithm
    if (algorithm === 'kmeans') {
        params.n_clusters = parseInt($('#kmeans-k').val());
    } else if (algorithm === 'hierarchical') {
        params.n_clusters = parseInt($('#hierarchical-k').val());
    } else if (algorithm === 'dbscan') {
        params.eps = parseFloat($('#dbscan-eps').val());
        params.min_samples = parseInt($('#dbscan-min-samples').val());
    }
    
    showLoading();
    
    // Send clustering request
    $.ajax({
        url: `${API_BASE_URL}/cluster`,
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            algorithm: algorithm,
            params: params
        }),
        success: function(response) {
            hideLoading();
            
            if (response.success) {
                // Display clustering results
                displayClusteringResults(response.result);
                
                // Move to next step
                currentStep = 4;
                updateStepIndicators();
            } else {
                showError(response.error || 'Error performing clustering.');
            }
        },
        error: function(xhr) {
            hideLoading();
            showError('Error performing clustering: ' + (xhr.responseJSON?.error || xhr.statusText));
        }
    });
}

/**
 * Displays clustering results including visualization and profiles
 */
function displayClusteringResults(result) {
    // Display visualization
    const visualizationContainer = $('#visualization-container');
    visualizationContainer.empty();
    visualizationContainer.append(`
        <h5 class="text-center">${result.algorithm} Visualization</h5>
        <div class="text-center">
            <img src="data:image/png;base64,${result.plot}" alt="Cluster Visualization" class="img-fluid">
        </div>
    `);
    
    // Display cluster summary
    const summaryContainer = $('#cluster-summary');
    summaryContainer.empty();
    
    // Basic info
    let summaryHtml = `
        <div class="cluster-info">
            <p><strong>Algorithm:</strong> ${result.algorithm}</p>
            <p><strong>Number of Clusters:</strong> ${result.n_clusters}</p>
    `;
    
    // Add silhouette score if available
    if (result.silhouette_score !== undefined) {
        const score = result.silhouette_score.toFixed(3);
        const scoreQuality = getScoreQuality(result.silhouette_score);
        summaryHtml += `
            <p>
                <strong>Silhouette Score:</strong> ${score} 
                <span class="badge bg-${scoreQuality.color}">${scoreQuality.label}</span>
            </p>
        `;
    }
    
    // Add noise points for DBSCAN
    if (result.n_noise !== undefined) {
        summaryHtml += `<p><strong>Noise Points:</strong> ${result.n_noise}</p>`;
    }
    
    summaryHtml += `</div>`;
    
    // Add cluster distribution
    summaryHtml += `<h6>Cluster Distribution</h6>`;
    
    // Count cluster sizes
    const clusters = result.clusters;
    const clusterCounts = {};
    clusters.forEach(cluster => {
        clusterCounts[cluster] = (clusterCounts[cluster] || 0) + 1;
    });
    
    // Generate distribution chart (simple bar representation)
    summaryHtml += `<div class="mb-3">`;
    Object.keys(clusterCounts).sort((a, b) => parseInt(a) - parseInt(b)).forEach(cluster => {
        const count = clusterCounts[cluster];
        const percentage = (count / clusters.length * 100).toFixed(1);
        const label = cluster === "-1" ? "Noise" : `Cluster ${cluster}`;
        const color = getClusterColor(parseInt(cluster));
        
        summaryHtml += `
            <div class="mb-2">
                <div class="d-flex justify-content-between mb-1">
                    <span>${label}</span>
                    <span>${count} (${percentage}%)</span>
                </div>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${percentage}%; background-color: ${color};"
                        aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </div>
        `;
    });
    summaryHtml += `</div>`;
    
    summaryContainer.append(summaryHtml);
    
    // Display cluster profiles
    const profilesContainer = $('#cluster-profiles-container');
    profilesContainer.empty();
    
    if (result.cluster_profiles) {
        // Create tabs for each cluster
        let tabsHtml = `
            <ul class="nav nav-tabs" id="cluster-tabs" role="tablist">
        `;
        
        let tabContentsHtml = `
            <div class="tab-content mt-2" id="cluster-contents">
        `;
        
        // Sort clusters numerically
        const clusterIds = Object.keys(result.cluster_profiles).sort((a, b) => {
            // Put noise cluster (-1) at the end
            if (a === "-1") return 1;
            if (b === "-1") return -1;
            return parseInt(a) - parseInt(b);
        });
        
        clusterIds.forEach((clusterId, index) => {
            const isActive = index === 0 ? 'active' : '';
            const label = clusterId === "-1" ? "Noise" : `Cluster ${clusterId}`;
            const color = getClusterColor(parseInt(clusterId));
            
            // Tab
            tabsHtml += `
                <li class="nav-item" role="presentation">
                    <button class="nav-link ${isActive}" id="cluster-${clusterId}-tab" data-bs-toggle="tab" 
                        data-bs-target="#cluster-${clusterId}" type="button" role="tab" 
                        aria-controls="cluster-${clusterId}" aria-selected="${index === 0}">
                        <span class="tag" style="background-color: ${color};">${label}</span>
                    </button>
                </li>
            `;
            
            // Tab content
            tabContentsHtml += `
                <div class="tab-pane fade show ${isActive}" id="cluster-${clusterId}" role="tabpanel" 
                    aria-labelledby="cluster-${clusterId}-tab">
                    <div class="table-container">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Value</th>
                                    <th>Compared to Average</th>
                                </tr>
                            </thead>
                            <tbody>
            `;
            
            // Get cluster profile data and calculate overall averages
            const clusterProfile = result.cluster_profiles[clusterId];
            const profileEntries = Object.entries(clusterProfile);
            
            // Calculate overall averages
            const allAverages = {};
            clusterIds.forEach(id => {
                const profile = result.cluster_profiles[id];
                Object.entries(profile).forEach(([feature, value]) => {
                    if (!allAverages[feature]) {
                        allAverages[feature] = {sum: 0, count: 0};
                    }
                    allAverages[feature].sum += value;
                    allAverages[feature].count++;
                });
            });
            
            // Convert to averages
            Object.keys(allAverages).forEach(feature => {
                allAverages[feature] = allAverages[feature].sum / allAverages[feature].count;
            });
            
            // Sort features by deviation from average (most distinctive first)
            profileEntries.sort((a, b) => {
                const [featureA, valueA] = a;
                const [featureB, valueB] = b;
                
                const deviationA = Math.abs(valueA - allAverages[featureA]) / allAverages[featureA];
                const deviationB = Math.abs(valueB - allAverages[featureB]) / allAverages[featureB];
                
                return deviationB - deviationA;
            });
            
            // Add rows for each feature
            profileEntries.forEach(([feature, value]) => {
                if (feature === 'Cluster') return; // Skip the cluster column itself
                
                const avg = allAverages[feature];
                const diff = ((value - avg) / avg * 100).toFixed(1);
                let diffHtml = '';
                
                if (diff > 5) {
                    diffHtml = `<span class="text-success">+${diff}%</span>`;
                } else if (diff < -5) {
                    diffHtml = `<span class="text-danger">${diff}%</span>`;
                } else {
                    diffHtml = `<span class="text-muted">${diff}%</span>`;
                }
                
                tabContentsHtml += `
                    <tr>
                        <td>${feature}</td>
                        <td>${typeof value === 'number' ? value.toFixed(2) : value}</td>
                        <td>${diffHtml}</td>
                    </tr>
                `;
            });
            
            tabContentsHtml += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        });
        
        tabsHtml += `</ul>`;
        tabContentsHtml += `</div>`;
        
        profilesContainer.append(tabsHtml + tabContentsHtml);
    } else {
        profilesContainer.html('<p>No profile information available.</p>');
    }
}

/**
 * Gets color for a cluster ID
 */
function getClusterColor(clusterId) {
    const colors = [
        '#3366cc', '#dc3912', '#ff9900', '#109618', '#990099',
        '#0099c6', '#dd4477', '#66aa00', '#b82e2e', '#316395',
        '#994499', '#22aa99', '#aaaa11', '#6633cc', '#e67300'
    ];
    
    // Noise is gray
    if (clusterId === -1) {
        return '#aaaaaa';
    }
    
    return colors[clusterId % colors.length];
}

/**
 * Gets quality label and color for silhouette score
 */
function getScoreQuality(score) {
    if (score >= 0.7) {
        return {label: 'Excellent', color: 'success'};
    } else if (score >= 0.5) {
        return {label: 'Good', color: 'primary'};
    } else if (score >= 0.3) {
        return {label: 'Fair', color: 'warning'};
    } else {
        return {label: 'Poor', color: 'danger'};
    }
}

/**
 * Exports clustering results as CSV
 */
function exportResults() {
    // Directly download file
    window.location.href = `${API_BASE_URL}/export`;
}

/**
 * Exports dataset as Weka ARFF file
 */
function exportWeka() {
    // Directly download file
    window.location.href = `${API_BASE_URL}/generate_weka`;
}

/**
 * Restarts the application
 */
function restartApplication() {
    // Reset global variables
    currentStep = 1;
    features = [];
    selectedFeatures = [];
    optimalK = null;
    
    // Reset UI elements
    $('#file-input').val('');
    $('#feature-selector').empty();
    $('#data-preview-table thead tr').html('<th>#</th>');
    $('#data-preview-table tbody').empty();
    $('#data-stats').html('<p>Preprocess data to see statistics</p>');
    $('#visualization-container').html('<p class="text-center">Run clustering to see visualization</p>');
    $('#cluster-summary').html('<p>Run clustering to see results</p>');
    $('#cluster-profiles-container').html('<p>Run clustering to see profiles</p>');
    
    // Reset to first step
    updateStepIndicators();
}

/**
 * Shows loading spinner
 */
function showLoading() {
    $('#loading').show();
}

/**
 * Hides loading spinner
 */
function hideLoading() {
    $('#loading').hide();
}

/**
 * Shows error message
 */
function showError(message) {
    $('#error-message').text(message);
    $('#error-alert').show();
    
    // Auto-hide after 5 seconds
    setTimeout(function() {
        $('#error-alert').fadeOut();
    }, 5000);
}
