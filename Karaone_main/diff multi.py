import pandas as pd
import matplotlib.pyplot as plt
import os


# Define the path to your .pkl files
files = [r'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\Raw_data\ImaginedSpeechData\MM05\GAM\df_epochs_ica.pkl', r'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\Raw_data\ImaginedSpeechData\MM05\GAM\df_epochs_filtered.pkl', r'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\Raw_data\ImaginedSpeechData\MM05\GAM\df_epochs_raw.pkl']
titl = ['raw', 'filtered', 'ica']
base_dir = r'E:\MP\Major Project - ISR MATLAB Code\Major Project - ISR MATLAB Code\Raw_data\ImaginedSpeechData\MM05\GAM\2to45'

# Ensure base directory exists
os.makedirs(base_dir, exist_ok=True)

# Sampling rate
sampling_rate = 1024
# Function to plot the data
def plot_data(files):
    t=3
    
    # Load the .pkl file and store the trial numbers for each file
    data_files = [pd.read_pickle(file) for file in files]
    titles = data_files[0].columns.tolist()
    print(titles)
    print(len(titles))
    # Get the unique trial numbers from the first file (assuming all files have the same trials)
    trial_numbers = data_files[0].iloc[:, 2].unique()
    
    for t in range(3 , len(titles)):
        folder_name = f"{titles[t]}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        os.chdir(folder_path)
    
    # Example: Save an image (replace with your actual image data)
        #image_path = os.path.join(folder_path, f"{image_name}.png")
    # Loop through each trial number
        for trial_number in trial_numbers:
            # Create a figure with 3 subplots (one for each file)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
            
            for i, data in enumerate(data_files):
                # Filter data for the selected trial number
                trial_data = data[data.iloc[:, 2] == trial_number]
                trial_name = trial_data.iloc[0, 1]  # Second column is the trial name

                
                # Extract time, trial name, and data columns
                time = trial_data.iloc[:, 0]  # First column is time
                trial_data_values = trial_data.iloc[:, t].values 
                trial_data_values = -trial_data_values # Fourth column is the data
                # Plot the data for this file in the corresponding subplot
                axs[i].plot(time, trial_data_values)
                axs[i].set_title(f"{titl[i]}")
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel("EEG Data")
                axs[i].set_xlim([time.min(), time.max()])  # Adjust the x-axis to the time range

            # Adjust the layout and show the plot for the current trial number
            fig.suptitle(f"Person: MM05 channel: {titles[t]} Trial Number: {trial_number} {trial_name}", fontsize=16)
            plt.tight_layout()
            filename = f"{trial_number}"
            plt.savefig(filename, dpi=300)  # Save as a high-resolution image
            print(f"channel: {t-3}, {titles[t]}, Figure saved: {trial_number}")
            
            # Optionally display the plot
            #plt.show()

            # Close the figure to free up memory
            plt.close(fig)
# Plot data for all trials from the 3 files
plot_data(files)



