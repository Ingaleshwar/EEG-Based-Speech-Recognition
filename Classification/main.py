import sys
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QWidget,
    QDialog, QComboBox, QSpinBox, QTextEdit, QDesktopWidget, QPlainTextEdit
)
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import numpy as np
import SUPP_SCRIPTS as mds
import mne
from mne.preprocessing import ICA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import copy
import matlab.engine
from string import Template
import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.backend import clear_session
from scipy.io import loadmat
import scipy.io as spio
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import gc
import h5py

plt.rcParams['figure.figsize'] = (20, 100)
class InitialPopup(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Trial")
        self.resize(400, 200)

        layout = QVBoxLayout()

        # Dropdown for selecting an option
        dropdown_label = QLabel("Select Person:")
        dropdown_label.setAlignment(Qt.AlignCenter)
        dropdown_label.setFont(QFont("Arial", 12))
        layout.addWidget(dropdown_label)

        self.dropdown = QComboBox()
        #self.dropdown.addItems(['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02'])
        self.dropdown.addItems(['MM05', 'MM10', 'MM11', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02'])# 'MM08', 'MM09', 'MM12', 'MM14', 'MM15'])
        self.dropdown.setStyleSheet("text-align: center;")
        self.dropdown.setFont(QFont("Arial", 12))
        layout.addWidget(self.dropdown, alignment=Qt.AlignCenter)

        # SpinBox for numerical input (1-48)
        spinbox_label = QLabel("Enter Trial Number (1-48):")
        spinbox_label.setAlignment(Qt.AlignCenter)
        spinbox_label.setFont(QFont("Arial", 12))
        layout.addWidget(spinbox_label)

        self.spinbox = QSpinBox()
        self.spinbox.setRange(1, 48)
        self.spinbox.setFont(QFont("Arial", 12))
        layout.addWidget(self.spinbox, alignment=Qt.AlignCenter)

        # Submit Button
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit)
        submit_button.setFont(QFont("Arial", 12))
        layout.addWidget(submit_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.center_popup()

    def submit(self):
        self.selected_option = self.dropdown.currentText()
        self.selected_number = self.spinbox.value()
        self.accept()  # Close the dialog with "accept" status

    def center_popup(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)



class EEGWorkflowGUI(QWidget):
    def __init__(self, selected_option, selected_number):
        super().__init__()
        self.selected_option = selected_option
        self.selected_number = selected_number
        self.subject = None
        self.filtered = None
        self.ica_data = None
        self.components = None
        self.ica = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('EEG-based Speech Recognition')
        self.resize(1200, 900)

        self.setWindowIcon(QIcon('logo.ico'))

        layout = QVBoxLayout()

        # Header Section
        header = QVBoxLayout()

        title2 = QLabel("Demonstration of")
        title2.setFont(QFont("Arial", 14))
        title2.setAlignment(Qt.AlignCenter)

        title3 = QLabel("EEG-based Speech Recognition")
        title3.setFont(QFont("Arial", 24, QFont.Bold))
        title3.setStyleSheet("color: red;")
        title3.setAlignment(Qt.AlignCenter)

        guide1 = QLabel("under the guidance of")
        guide1.setFont(QFont("Arial", 10))
        guide1.setAlignment(Qt.AlignCenter)

        guide2 = QLabel("Dr. Veena Karjigi")
        guide2.setFont(QFont("Arial", 16))
        guide2.setStyleSheet("color: lime;")
        guide2.setAlignment(Qt.AlignCenter)

        dept = QLabel("Department of ECE.\nSIT, Tumakuru 03")
        dept.setFont(QFont("Arial", 12))
        dept.setAlignment(Qt.AlignCenter)

        header.addWidget(title2)
        header.addWidget(title3)
        header.addWidget(guide1)
        header.addWidget(guide2)
        header.addWidget(dept)

        # Names and USNs
        names_usns = QHBoxLayout()
        name_labels = [
            QLabel("Aditya Keshav Harikantra"),
            QLabel("Harsha M"),
            QLabel("Rahul Jain S V"),
            QLabel("Rohith Ingaleshwar")
        ]
        usn_labels = [
            QLabel("1SI21EC004"),
            QLabel("1SI21EC037"),
            QLabel("1SI21EC074"),
            QLabel("1SI21EC078")
        ]
        for name, usn in zip(name_labels, usn_labels):
            column = QVBoxLayout()
            name.setFont(QFont("Arial", 12))
            name.setAlignment(Qt.AlignCenter)
            usn.setFont(QFont("Arial", 10))
            usn.setAlignment(Qt.AlignCenter)
            column.addWidget(name)
            column.addWidget(usn)
            column.setSpacing(-40)  # Minimum spacing between name and USN
            names_usns.addLayout(column)

        # Display selected inputs
        inputs = QLabel(f"Selected Option: {self.selected_option}\nSelected Number: {self.selected_number}")
        inputs.setFont(QFont("Arial", 14))
        inputs.setAlignment(Qt.AlignCenter)

        # Buttons Section
        buttons_layout = QVBoxLayout()
        button_font = QFont("Arial", 14)  # Bigger font size for buttons

        buttons_layout.addWidget(QPushButton("Load Raw EEG Data", clicked=self.load_raw_eeg_data, font=button_font))
        buttons_layout.addWidget(QPushButton("Split Data Into Trials", clicked=self.split_data, font=button_font))

        row_1 = QHBoxLayout()
        row_1.addWidget(QPushButton("Plot Raw Signal", clicked=self.plot_raw_signal, font=button_font))
        row_1.addWidget(QPushButton("Plot PSD", clicked=self.plot_psd, font=button_font))
        buttons_layout.addLayout(row_1)

        buttons_layout.addWidget(QPushButton("Filter Signal (2-45 Hz)", clicked=self.filter_signal, font=button_font))

        row_2 = QHBoxLayout()
        row_2.addWidget(QPushButton("Plot Filtered Signal", clicked=self.plot_filtered_signal, font=button_font))
        row_2.addWidget(QPushButton("Plot PSD of Filtered Signal", clicked=self.plot_filtered_psd, font=button_font))
        buttons_layout.addLayout(row_2)

        buttons_layout.addWidget(QPushButton("Decompose Filtered Signal into Components", clicked=self.decompose_signal, font=button_font))

        row_3 = QHBoxLayout()
        row_3.addWidget(QPushButton("Auto Artifact Rejection", clicked=self.auto_artifact_rejection, font=button_font))
        row_3.addWidget(QPushButton("Inspect Components", clicked=self.inspect_components, font=button_font))
        buttons_layout.addLayout(row_3)

        buttons_layout.addWidget(QPushButton("Reconstruct Signal", clicked=self.reconstruct_signal, font=button_font))
        #buttons_layout.addWidget(QPushButton("Load Model", clicked=self.load_model, font=button_font))

        row_4 = QHBoxLayout()
        row_4.addWidget(QPushButton("SVM Prediction", clicked=self.svm_prediction, font=button_font))
        row_4.addWidget(QPushButton("LSTM Prediction", clicked=self.lstm_prediction, font=button_font))
        buttons_layout.addLayout(row_4)



        # Output boxes
        output_row = QHBoxLayout()
        self.predicted_word_box = QTextEdit()
        self.predicted_word_box.setReadOnly(True)
        self.predicted_word_box.setFixedHeight(50)
        self.predicted_word_box.setPlaceholderText("Predicted Word")

        self.actual_word_box = QTextEdit()
        self.actual_word_box.setReadOnly(True)
        self.actual_word_box.setFixedHeight(50)
        self.actual_word_box.setPlaceholderText("Actual Word")

        output_row.addWidget(self.predicted_word_box)
        output_row.addWidget(self.actual_word_box)

        # Output and Terminal Section
        output_terminal_layout = QVBoxLayout()

        # Terminal-like output
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFont(QFont("Courier", 10))
        self.terminal.setPlaceholderText("Terminal Output")
        self.terminal.setFixedHeight(150)
        output_terminal_layout.addLayout(output_row)
        output_terminal_layout.addWidget(self.terminal)

        # Combine everything
        layout.addLayout(header)
        layout.addLayout(names_usns)
        layout.addWidget(inputs)
        layout.addLayout(buttons_layout)
        layout.addLayout(output_terminal_layout)

        self.setLayout(layout)

    # Function Stubs
    def load_raw_eeg_data(self):
        subject = selected_option
        PATH_TO_DATA = "C:\\Users\\Rahul Jain\\Desktop\\major_demo\\data\\"
        mds.Dataset(subject)
        for subject in mds.Dataset.registry:
            print("Working on Subject: " + subject.name)
            print('Loading data..')
            subject.load_data(PATH_TO_DATA, raw=True)
            print('Data Loaded.')
            
            self.log_terminal("Data Loaded.")
            print(subject.eeg_data.info)

            subject.eeg_data.load_data()
            self.log_terminal("Plotting Input Data")
            subject.eeg_data.plot(color='darkblue')
            self.subject = subject
        self.log_terminal("Load Raw EEG Data clicked")

    def split_data(self):
        subject = self.subject
        raw = subject.eeg_data.copy()
        #filtered = subject.eeg_data.copy()
        #ica_data = subject.eeg_data.copy()

        # Bandpass filter between 1Hz and 50Hz (also removes power line noise ~60Hz)
        # filtered.filter(None, 45., fir_design='firwin')
        # filtered.filter(2., None, fir_design='firwin')
        # ica_data.filter(None, 45., fir_design='firwin')
        # ica_data.filter(2., None, fir_design='firwin')
        events = copy.deepcopy(subject.epoch_inds['thinking_inds'])
        events = np.reshape(events, (events.shape[1], 1))
        prompts = []
        for event in events:
            prompts.append(event[0][0])

        i = 0
        for prompt in prompts:
            prompt[1] = 0
            prompts[i] = np.append(prompt, np.array(subject.prompts[5][0][i][0]))
            i += 1

        prompts = np.asarray(prompts)

        # All prompts need to be int format
        prompts = np.where(prompts == '/iy/', 0, prompts)
        prompts = np.where(prompts == '/uw/', 1, prompts)
        prompts = np.where(prompts == '/piy/', 2, prompts)
        prompts = np.where(prompts == '/tiy/', 3, prompts)
        prompts = np.where(prompts == '/diy/', 4, prompts)
        prompts = np.where(prompts == '/m/', 5, prompts)
        prompts = np.where(prompts == '/n/', 6, prompts)
        prompts = np.where(prompts == 'pat', 7, prompts)
        prompts = np.where(prompts == 'pot', 8, prompts)
        prompts = np.where(prompts == 'knew', 9, prompts)
        prompts = np.where(prompts == 'gnaw', 10, prompts)

        #sanity check
        prompts = prompts.astype(int)
        print('MNE event_id array for epoching:')
        # Create event identification dictionary
        event_id = {'/iy/': 0, '/uw/': 1, '/piy/': 2, '/tiy/': 3, '/diy/': 4, '/m/': 5,
                    '/n/': 6, 'pat': 7, 'pot': 8, 'knew': 9, 'gnaw': 10}
        epochs_raw = mne.Epochs(raw, prompts, event_id, tmin=-0.01, tmax=5.0, baseline=None, preload=True)
        epochs_raw.plot()
        del events, prompts
        self.log_terminal("Split Data Into Trials clicked")

    def plot_raw_signal(self):
        subject = self.selected_option
        file = rf'C:\Users\Rahul Jain\Desktop\major_demo\data\{subject}\plot_files_raw.pkl'

        # List of indices for the columns you want to plot
        t_values = [25, 39, 40, 28, 26]

        # Load the .pkl file and get the DataFrame
        data = pd.read_pickle(file)
        titles = data.columns.tolist()

        # Get the unique trial numbers from the file
        trial_numbers = data.iloc[:, 2].unique()

        # Select a trial number (e.g., trial_number = 1)
        trial_number = self.selected_number

        # Create a 3x2 grid of subplots (total 6, 1 of them will remain unused)
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns

        # Flatten the axs array to simplify indexing in the loop
        axs = axs.flatten()

        # Loop through each of the columns you want to plot
        for i, t in enumerate(t_values):
            # Filter data for the selected trial number
            trial_data = data[data.iloc[:, 2] == trial_numbers[trial_number - 1]] 
            condition = trial_data.iloc[0, 1] # Adjusted for trial number
            self.actual_word_box.setText(f'Actual Prompt of the Trial: {condition}')
            # Extract time and data for the current column index (t)
            time = trial_data.iloc[:, 0]  # First column is time
            trial_data_values = trial_data.iloc[:, t].values  # Get the data for the column at index t
            
            # Plot the data on the corresponding subplot
            axs[i].plot(time, trial_data_values)
            axs[i].set_title(f"Channel: {titles[t]} (Index: {t})")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("EEG Data")
            axs[i].set_xlim([time.min(), time.max()])  # Adjust the x-axis to the time range

        # Remove the last (empty) subplot, which will be left unused
        fig.delaxes(axs[5])

        # Center the last plot in the third row by adjusting the position of the last subplot
        axs[4].set_position([0.4, 0.05, 0.2, 0.2])  # Manually adjust the position of the 5th subplot

        # Adjust the layout and show the plot
        fig.suptitle(f"Person: {subject} Trial Number: {trial_number}", fontsize=16)
        plt.tight_layout()

        # Optionally, save the plot or display it
        # filename = f"{trial_number}.png"
        # plt.savefig(filename, dpi=300)  # Save as a high-resolution image

        # Display the plot
        plt.show()

    def plot_psd(self): 
        subject = self.subject
        
        subject.eeg_data.plot_psd(area_mode='range', tmax=10.0)
        # #Plot the raw eeg psd with Fmax of 50Hz

        self.log_terminal("Plot PSD clicked")


    def filter_signal(self): 
        subject = self.subject
        events = copy.deepcopy(subject.epoch_inds['thinking_inds'])
        events = np.reshape(events, (events.shape[1], 1))
        prompts = []
        for event in events:
            prompts.append(event[0][0])

        i = 0
        for prompt in prompts:
            prompt[1] = 0
            prompts[i] = np.append(prompt, np.array(subject.prompts[5][0][i][0]))
            i += 1

        prompts = np.asarray(prompts)

        # All prompts need to be int format
        prompts = np.where(prompts == '/iy/', 0, prompts)
        prompts = np.where(prompts == '/uw/', 1, prompts)
        prompts = np.where(prompts == '/piy/', 2, prompts)
        prompts = np.where(prompts == '/tiy/', 3, prompts)
        prompts = np.where(prompts == '/diy/', 4, prompts)
        prompts = np.where(prompts == '/m/', 5, prompts)
        prompts = np.where(prompts == '/n/', 6, prompts)
        prompts = np.where(prompts == 'pat', 7, prompts)
        prompts = np.where(prompts == 'pot', 8, prompts)
        prompts = np.where(prompts == 'knew', 9, prompts)
        prompts = np.where(prompts == 'gnaw', 10, prompts)

        #sanity check
        prompts = prompts.astype(int)
        print('MNE event_id array for epoching:')

        # Create event identification dictionary
        event_id = {'/iy/': 0, '/uw/': 1, '/piy/': 2, '/tiy/': 3, '/diy/': 4, '/m/': 5,
                    '/n/': 6, 'pat': 7, 'pot': 8, 'knew': 9, 'gnaw': 10}
        
        filtered = subject.eeg_data.copy()
        #ica_data = subject.eeg_data.copy()

        # Bandpass filter between 1Hz and 50Hz (also removes power line noise ~60Hz)
        filtered.filter(None, 45., fir_design='firwin')
        filtered.filter(2., None, fir_design='firwin')

        self.filtered = filtered
        epochs_filtered = mne.Epochs(filtered, prompts, event_id, tmin=-0.01, tmax=5.0, baseline=None, preload=True)
        epochs_filtered.plot()
        self.ica_data = epochs_filtered

        self.log_terminal("Filter Signal clicked")


    def plot_filtered_signal(self):
        subject = self.selected_option
        file = rf'C:\Users\Rahul Jain\Desktop\major_demo\data\{subject}\plot_files_filtered.pkl'

        # List of indices for the columns you want to plot
        t_values = [25, 39, 40, 28, 26]

        # Load the .pkl file and get the DataFrame
        data = pd.read_pickle(file)
        titles = data.columns.tolist()

        # Get the unique trial numbers from the file
        trial_numbers = data.iloc[:, 2].unique()

        # Select a trial number (e.g., trial_number = 1)
        trial_number = self.selected_number

        # Create a 3x2 grid of subplots (total 6, 1 of them will remain unused)
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns

        # Flatten the axs array to simplify indexing in the loop
        axs = axs.flatten()

        # Loop through each of the columns you want to plot
        for i, t in enumerate(t_values):
            # Filter data for the selected trial number
            trial_data = data[data.iloc[:, 2] == trial_numbers[trial_number - 1]] 
            condition = trial_data.iloc[0, 1] # Adjusted for trial number
            self.actual_word_box.setText(f'Actual Prompt of the Trial: {condition}')
            # Extract time and data for the current column index (t)
            time = trial_data.iloc[:, 0]  # First column is time
            trial_data_values = trial_data.iloc[:, t].values  # Get the data for the column at index t
            
            # Plot the data on the corresponding subplot
            axs[i].plot(time, trial_data_values)
            axs[i].set_title(f"Channel: {titles[t]} (Index: {t})")
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel("EEG Data")
            axs[i].set_xlim([time.min(), time.max()])  # Adjust the x-axis to the time range

        # Remove the last (empty) subplot, which will be left unused
        fig.delaxes(axs[5])

        # Center the last plot in the third row by adjusting the position of the last subplot
        axs[4].set_position([0.4, 0.05, 0.2, 0.2])  # Manually adjust the position of the 5th subplot

        # Adjust the layout and show the plot
        fig.suptitle(f"Person: {subject} Trial Number: {trial_number}", fontsize=16)
        plt.tight_layout()

        # Optionally, save the plot or display it
        # filename = f"{trial_number}.png"
        # plt.savefig(filename, dpi=300)  # Save as a high-resolution image

        # Display the plot
        plt.show()
        self.log_terminal("Plot Filtered Signal clicked")
    
    def plot_filtered_psd(self):
        filtered = self.filtered
        filtered.plot_psd(area_mode='range', tmax=10.0)

        self.log_terminal("Plot PSD of Filtered Signal clicked")
    
    def decompose_signal(self):
        epochs_ica = self.ica_data
        method = 'fastica'

        # Choose other parameters
        n_components = 62  # if float, select n_components by explained variance of PCA
        decim = 3  #if needed, decimate the time points for efficiency
        random_state = 23

        ica = ICA(n_components=n_components, method=method, random_state=random_state)
        # Typical threshold for rejection as it is undesireable to fit to these extreme values
        reject = dict(mag=5e-12, grad=4000e-13)

        # fit ICA
        ica.fit(epochs_ica, reject=reject)
        self.ica = ica
        self.components = epochs_ica
        self.log_terminal("Decompose Filtered Signal into Components clicked")


    def auto_artifact_rejection(self):
        
        # Start the MATLAB engine
        eng = matlab.engine.start_matlab()

        # Now, you can proceed with your original code to load and process the EEG data
        matlab_code = r"""
        [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
        EEG = pop_loadset('filename', 'processed_data_AA_with_ICA15.set', 'filepath', 'C:/Users/Rahul Jain/Downloads/Skull/15/');
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
        pop_selectcomps(EEG, [1:62]);
        eeglab redraw;
        close all;
        clc;
        """

        # Execute your EEG data processing code
        eng.eval(matlab_code, nargout=0)


        # Optionally, you can close the MATLAB engine after running the code
        eng.quit()

        self.log_terminal("Auto Artifact Rejection clicked")
    
    def inspect_components(self):
        subject = self.selected_option
        eng = matlab.engine.start_matlab()
        string2 = Template(r"""
        [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
        EEG = pop_loadset('filename','processed_data_AA_with_ICA.set','filepath','C:\Users\Rahul Jain\Desktop\major_demo\data\$subject\');
        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
        EEG = pop_iclabel(EEG, 'default');
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
        addpath(genpath('C:\Users\Rahul Jain\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\EEGLAB\plugins\ICLabel\viewprops'));
        EEG = pop_viewprops(EEG, 0, 1:62, {'freqrange', [2 64]}, {}, 1, 'ICLabel');
        EEG = ALLEEG(CURRENTSET);
        eeglab redraw;

        uiwait(gcf);
        """)
        matlab_code = string2.substitute(subject=subject)

        # Now, you can proceed with your original code to load and process the EEG data
        # Execute your EEG data processing code
        eng.eval(matlab_code, nargout=0)


        # Optionally, you can close the MATLAB engine after running the code
        eng.quit()
        self.log_terminal("Inspect Components clicked")
    
    def reconstruct_signal(self):
        ica = self.ica
        n_components = 62  # if float, select n_components by explained variance of PCA
        decim = 3  #if needed, decimate the time points for efficiency
        random_state = 23
        epochs_ica = self.components
        ica.apply(epochs_ica, n_pca_components=n_components, exclude=ica.exclude)
        self.log_terminal("Reconstruct Signal clicked")
    
    def load_model(self):
        self.log_terminal("Load Model clicked")
    
    def svm_prediction(self):
        subject = self.selected_option
        trial = self.selected_number
        model_filename = r'C:\Users\Rahul Jain\Desktop\major_demo\data\best_svm_model.joblib'
        path = r'C:\Users\Rahul Jain\Desktop\major_demo\data'
        # Set ICA flag and participant file paths
        ica = ''
        participant_files = [
            rf'{path}\{subject}\all_features_{ica}ICA.mat'
        ]

        # Initialize lists to store consolidated data
        all_eeg_features = []
        all_prompts = []

        # Define feature range for 'i' and 'j' target prompts
        target_prompts = ['gnaw', 'knew', 'pot', 'pat']

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
        X_train, X_test, y_train, y_test = train_test_split(all_eeg_features, all_prompts, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the SVM model
        svm_model = SVC(kernel='rbf', C=10, random_state=42)
        svm_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = svm_model.predict(X_test)

        C = svm_model.predict(X_test)
        self.predicted_word_box.setText(f'SVM :{target_prompts[C[trial]]}')
        self.log_terminal("SVM Prediction clicked")
    
    def lstm_prediction(self):
        def loadmat(filename):
            def _check_keys(d):
                '''
                checks if entries in dictionary are mat-objects. If yes
                todict is called to change them to nested dictionaries
                '''
                for key in d:
                    if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                        d[key] = _todict(d[key])
                return d

            def _has_struct(elem):
                """Determine if elem is an array and if any array item is a struct"""
                return isinstance(elem, np.ndarray) and any(isinstance(
                            e, spio.matlab.mio5_params.mat_struct) for e in elem)

            def _todict(matobj):
                '''
                A recursive function which constructs from matobjects nested dictionaries
                '''
                d = {}
                for strg in matobj._fieldnames:
                    elem = matobj.__dict__[strg]
                    if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                        d[strg] = _todict(elem)
                    elif _has_struct(elem):
                        d[strg] = _tolist(elem)
                    else:
                        d[strg] = elem
                return d

            def _tolist(ndarray):
                '''
                A recursive function which constructs lists from cellarrays
                (which are loaded as numpy ndarrays), recursing into the elements
                if they contain matobjects.
                '''
                elem_list = []
                for sub_elem in ndarray:
                    if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                        elem_list.append(_todict(sub_elem))
                    elif _has_struct(sub_elem):
                        elem_list.append(_tolist(sub_elem))
                    else:
                        elem_list.append(sub_elem)
                return elem_list
            data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
            return _check_keys(data)

        """Helper function to truncate dataframes to a specified shape - usefull to reduce all EEG trials to the same number
        of time stamps.
        """
        def truncate(arr, shape):
            desired_size_factor = np.prod([n for n in shape if n != -1])
            if -1 in shape:  # implicit array size
                desired_size = arr.size // desired_size_factor * desired_size_factor
            else:
                desired_size = desired_size_factor
            return arr.flat[:desired_size].reshape(shape)

        def load_single_trial(subject, trial_index):
            """
            Load EEG data and select a single trial by index.

            Parameters:
            - filepath: Path to the `.mat` file containing EEG data.
            - trial_index: Index of the trial to be selected.

            Returns:
            - A numpy array of shape (timesteps, features) for the selected trial.
            """
            PATH = "C:\KARA_ONE_Data\ImaginedSpeechData\\"

            print("Working on Subject: " + subject)

            print("Loading .set data")
            """ Load EEG data with loadmat() function"""
            SubjectData = loadmat(PATH + subject + '\\EEG_data.mat')

            print("Setting up dataframes")
            """ Setup target and EEG dataframes"""
            targets = pd.DataFrame(SubjectData['EEG_Data']['prompts'])
            targets.columns = ['prompt']

            sequences = pd.DataFrame(SubjectData['EEG_Data']['activeEEG'])
            sequences.columns = ['trials']
            #print(targets)

            EEG = pd.concat([sequences.reset_index(drop=True),targets.reset_index(drop=True)], axis=1)

            labels = ['gnaw', 'pat', 'knew', 'pot']


            EEG = EEG.loc[EEG['prompt'].isin(labels)]

            EEG = EEG.reset_index(drop=True)
            #print(EEG['trials'])


            sequences = pd.DataFrame(EEG['trials'])
            targets = pd.DataFrame(EEG['prompt'])

            sequences = sequences.loc[trial_index + 1, 'trials']
            targets = targets.loc[trial_index + 1, 'prompt']

            seq = np.asarray(sequences)
            seq = seq.transpose()


            sequences = seq  # Transpose to match the (timesteps, features) format
            sequences = np.asarray(sequences)

            # #find minimum length of all the trials present in both test and train trials
            # min_ln = min(min(i.shape for i in sequences)[0], min(i.shape for i in sequences)[0])

            # #reduce all trials down to common length set by min_ln
            # for arr in [sequences, targets]:
            #     i=0
            #     for trial in arr:
            #         arr[i] = truncate(trial, (min_ln, 62))
            #         i = i+1

            #for LSTM model we need data in a 3D array, (,
            sequences = np.rollaxis(np.dstack(sequences), -1)

            return sequences, targets


        def test_single_trial(model_path, trial_data, target_labels):
            """
            Test a single EEG trial with the trained LSTM model.
            
            Parameters:
            - model_path: Path to the saved LSTM model.
            - trial_data: A numpy array of shape (timesteps, features) representing a single trial.
            - target_labels: List of target labels corresponding to the model's output classes.
            
            Returns:
            - Predicted label for the trial.
            """
            # Ensure trial data is in the correct 3D shape: (1, timesteps, features)

            # Load the trained model
            model = load_model(model_path)
            
            # Predict the class
            prediction = model.predict(trial_data)
            predicted_class = np.argmax(prediction, axis=-1)[0]

            # Map predicted class to label
            predicted_label = target_labels[predicted_class]
            
            clear_session()  # Clear the session to free memory
            
            return predicted_label
        PATH = "C:\\KARA_ONE_Data\\ImaginedSpeechData\\"
        subject = self.selected_option
        trial_index = self.selected_number
        # Load the single trial data
        trial_data, word = load_single_trial(subject, trial_index)
        labels = ['gnaw', 'pat', 'knew', 'pot']
        saved_model_path = PATH + subject + '\\lstm_model' + '\\lstm_vanilla_model.keras'
        predicted_label = test_single_trial(saved_model_path, trial_data, labels)
        self.predicted_word_box.setText(f"LSTM {predicted_label}")
        self.log_terminal("LSTM Prediction clicked")

    def log_terminal(self, message):
        
        self.terminal.appendPlainText(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Show initial popup
    popup = InitialPopup()
    if popup.exec_() == QDialog.Accepted:
        selected_option = popup.selected_option
        selected_number = popup.selected_number

        # Launch the main GUI
        gui = EEGWorkflowGUI(selected_option, selected_number)
        gui.show()
        sys.exit(app.exec_())
