import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import pandas as pd
model_filename = r'C:\Users\adity\Desktop\SVM_boss\Plots\best_svm_model.joblib'
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Set ICA flag and participant file paths
ica = 'no'
participant_files = [
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM05\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM08\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM09\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM10\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM11\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM12\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM14\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM15\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM16\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM18\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM19\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM20\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\MM21\all_features_{ica}ICA.mat',
    rf'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\data\P02\all_features_{ica}ICA.mat',
]

# Initialize lists to store consolidated data
all_eeg_features = []
all_prompts = []

# Define feature range for 'i' and 'j' target prompts
target_prompts = ['gnaw', 'knew', 'pot', 'pat']

# Best accuracy initialization
best_accuracy = 0
best_i_value = None
best_j_value = None
best_model = None
best_X_train, best_X_test, best_y_train, best_y_test = None, None, None, None

# Iterate through each participant file
for file_path in participant_files:
    # Load the participant's data
    data = sio.loadmat(file_path)
    print(f"Processing {file_path}")
    
    # Extract relevant data for this participant
    thinking_feats = data['all_features']['eeg_features'][0][0]['thinking_feats']
    feature_labels = data['all_features']['feature_labels'][0][0].flatten()
    prompts = data['all_features']['prompts'][0][0].flatten()

    # Ensure `thinking_feats` is a list of NumPy arrays
    thinking_feats = [np.array(trial) for trial in thinking_feats]

    # Iterate over i and j in the specified range
    for i in range(1, 5):  # i from 1 to 4
        for j in range(1, 4):  # j from 1 to 3
            i_str = str(i)
            j_str = str(j)
            m_str = str(3)  # Adjust if necessary

            # Selected features based on the value of i and j
            selected_features = [
                "Mean:W" + i_str, "dMean:W" + i_str, "Energy:W" + i_str, "Absmin:W" + i_str,
                "EHF:W" + m_str, "CurveLength:W" + m_str, "Absmean:W" + m_str, "Max-Min:W" + m_str
            ]

            # Map feature labels to their indices
            selected_feature_indices = [i for i, label in enumerate(feature_labels) if label in selected_features]

            # Filter relevant prompts and corresponding EEG data
            selected_indices = [i for i, prompt in enumerate(prompts) if prompt in target_prompts]
            filtered_eeg_features = [thinking_feats[0][0][0][i] for i in selected_indices]
            filtered_prompts = [prompts[i] for i in selected_indices]

            # Extract the selected features
            X = np.array([trial[:, selected_feature_indices] for trial in filtered_eeg_features])

            # Flatten the channel dimension
            X = X.reshape(X.shape[0], -1)

            # Convert prompts to numerical labels
            target_mapping = {prompt: idx for idx, prompt in enumerate(target_prompts)}
            filtered_prompts = np.array([item[0] for item in filtered_prompts])
            y = np.array([target_mapping[prompt] for prompt in filtered_prompts])

            # Append the features and labels to the consolidated lists
            all_eeg_features.append(X)
            all_prompts.append(y)

# Convert the lists into NumPy arrays
all_eeg_features = np.vstack(all_eeg_features)
all_prompts = np.hstack(all_prompts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(all_eeg_features, all_prompts, test_size=0.025, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='rbf', C=10, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
current_accuracy = np.mean(y_pred == y_test)

# Check if this is the best accuracy so far
if current_accuracy > best_accuracy:
    best_accuracy = current_accuracy
    best_i_value = i_str
    best_j_value = j_str
    best_model = svm_model
    best_X_train, best_X_test, best_y_train, best_y_test = X_train, X_test, y_train, y_test

# Print best accuracy and i, j values
print(f"Best accuracy: {best_accuracy} with i={best_i_value} and j={best_j_value}")

# Print the classification report for the best model
y_pred_best = best_model.predict(best_X_test)
print(classification_report(best_y_test, y_pred_best, target_names=target_prompts))


# model_filename = 'best_svm_model.pkl'
# joblib.dump(best_model, model_filename)
# print(f"Best model saved as {model_filename}")

joblib.dump(best_model, model_filename)
print(f"Best model saved as {model_filename}")


# Save the results to an Excel sheet
trial_numbers = list(range(1, len(best_X_test) + 1))  # Generate trial numbers
actual_words = [target_prompts[label] for label in best_y_test]  # Map numerical labels to actual words
predicted_words = [target_prompts[label] for label in y_pred_best]  # Map numerical predictions to words

# Create a DataFrame
results_df = pd.DataFrame({
    'Trial Number': trial_numbers,
    'Actual Word': actual_words,
    'Predicted Word': predicted_words
})

# Define the Excel file path
excel_file_path = r'C:\Users\adity\Desktop\SVM_boss\xcel.xlsx'

# Save the DataFrame to an Excel file
results_df.to_excel(excel_file_path, index=False, engine='openpyxl')

print(f"Results saved to {excel_file_path}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(best_y_test, y_pred_best)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=target_prompts)

# Plot the confusion matrix
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')

# Save the confusion matrix as an image
conf_matrix_image_path = r'C:\Users\adity\Desktop\SVM_boss\confusion_matrix.png'
plt.savefig(conf_matrix_image_path)

print(f"Confusion matrix saved to {conf_matrix_image_path}")

# Show the plot (optional)
plt.show()

report = classification_report(best_y_test, y_pred_best, target_names=target_prompts)

# Create a figure to save the report
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed
ax.axis('off')  # Turn off the axes

# Add the classification report text to the figure
ax.text(0.01, 0.01, report, fontsize=12, fontfamily='monospace')  # Use monospace font for alignment

# Save the classification report as an image
classification_report_image_path = r'C:\Users\adity\Desktop\SVM_boss\classification_report.png'
plt.savefig(classification_report_image_path, bbox_inches='tight')

print(f"Classification report saved to {classification_report_image_path}")

# Optional: Display the image
plt.show()