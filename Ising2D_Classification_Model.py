import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from matplotlib.colors import ListedColormap
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
import pandas as pd
from keras.layers import Lambda
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input
from keras.utils import Sequence


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"TensorFlow GPU setup complete: {physical_devices[0]}")
    except Exception as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found")


# Variables used for file names
num_of_total_files = 2000
num_of_tr = 2000
num_of_ind = 2000
L = 512
saved = 2000

# We train the model on the extreme physical limits, which is lattice configurations at T = 0 and T = Infinity.
# These lattice configurations are not built using MCMC Simulations (this can't be done); instead we made an algorithm to generate them,
# knowing the fact that at T = 0 all spins with no exception are pointing in the same direction, 
# all aligned up or down. So we generated 1000 lattice configurations all up, and 1000 lattice configurations all down, so the 2000
# configuration are labeled ferromagnetic. Similarly knowing the fact that at T = Infinity, the average magnetization per spin is exactly
# zero, we gnerated 2000 lattice configurations which have exactly zero magnetization    
 
T_b = 0
T_A = "Inf"

# Parameters
batch_size = 32  # Adjust based on available memory
input_shape = (L, L, 1)
temperature = 1 # Scaling factor for the lambda layer. Using temperature = 1 is equivalent to having 
                # no lambda layer, which is what we ended up doing for the paper.  
learning_rate = 1e-4 if L <= 256 else 1e-5 # Adjust learning rate based on lattice size
label_names = ['Paramagnetic', 'Ferromagnetic']

# Custom data generator
class FileDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, input_shape):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.indexes = np.arange(len(self.file_paths))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.file_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        data = [np.load(file).reshape(self.input_shape) for file in batch_files]
        labels = np.array(batch_labels)
        return np.array(data), labels

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# Data
ferro_files = [os.path.join("Ferromagnetic", f) for f in os.listdir("Ferromagnetic")]
para_files = [os.path.join("Paramagnetic", f) for f in os.listdir("Paramagnetic")]
ferro_labels = [1] * len(ferro_files)
para_labels = [0] * len(para_files)

all_files = ferro_files + para_files
all_labels = ferro_labels + para_labels

train_files, test_files, train_labels, test_labels = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42
)

# Split training data into training and validation sets
train_files_split, val_files, train_labels_split, val_labels = train_test_split(
    train_files, train_labels, test_size=0.1, random_state=42)

# Create generators for training, validation, and testing
train_generator = FileDataGenerator(train_files_split, train_labels_split, batch_size, input_shape)
val_generator = FileDataGenerator(val_files, val_labels, batch_size, input_shape)
test_generator = FileDataGenerator(test_files, test_labels, batch_size, input_shape)

# Build the model. This is the model used in the BAL paper 2004.14341. 
# We add a lambda layer to potentially reduce the overffiting and the confidence of the model in its predictions; 
# however in this specific code we don't use this layer because we se the scaling factor to 1. 
model = Sequential([
    Input(shape=input_shape),
    Conv2D(64, kernel_size=(2, 2), strides=(2, 2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Lambda(lambda x: x / temperature),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Accuracy.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Loss.png')
    plt.show()

plot_history(history)

# Evaluate the model directly using the test generator
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Collect predictions and true labels batch-by-batch from the test generator
predictions = []
true_labels = []

# Process the test set batch-wise
for i in range(len(test_generator)):
    X_batch, y_batch = test_generator[i]  # Get batch of data and labels
    batch_predictions = model.predict(X_batch, verbose=0)  # Predict for the batch
    predictions.extend(batch_predictions)  # Append predictions for the batch
    true_labels.extend(y_batch)  # Append true labels

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy explicitly
test_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy (calculated): {test_accuracy}")

# Confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap=ListedColormap(['orange', 'blue']),
            xticklabels=label_names, yticklabels=label_names)
plt.title('2D Ising classification model - Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.show()
plt.savefig(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Confusion_Matrix.png')

# Generate a classification report
report = classification_report(true_labels, predicted_labels, target_names=label_names, output_dict=True)

# Save the classification report to a text file
report_filename = f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Classification_Report.txt'
with open(report_filename, 'w') as f:
    f.write("Classification Report:\n")
    f.write("-" * 60 + "\n")
    for label in label_names:
        metrics = report[label]
        f.write(f"{label:<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
        f.write(f"{'':<20} {metrics['precision']:<12.2f} {metrics['recall']:<12.2f} {metrics['f1-score']:<12.2f} {metrics['support']:<12}\n")
    f.write("-" * 60 + "\n")
    accuracy = report['accuracy']
    f.write(f"{'Accuracy:':<20} {accuracy:>7.2f}\n")

print(f"Classification report saved to: {report_filename}")

# Display classification report in the console
print("Classification Report:")
print("-" * 60)
for label in label_names:
    metrics = report[label]
    print(f"{label:<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print(f"{'':<20} {metrics['precision']:<12.2f} {metrics['recall']:<12.2f} {metrics['f1-score']:<12.2f} {metrics['support']:<12}")
print("-" * 60)
print(f"{'Accuracy:':<20} {report['accuracy']:>7.2f}")

# Clear memory
del test_generator, predictions, true_labels, confusion, report
gc.collect()













# Beginning of the Independent testing code.
# Function to calculate first derivative of average prediction with respect to T
def calculate_first_derivative(temps, avg_predictions, avg_errors):
    derivatives_of_AvgP = []
    errors_in_derivatives = []

    for i in range(len(temps)):
        if i == 0:  # Forward difference for the first point
            derivative = (avg_predictions[i+1] - avg_predictions[i]) / (temps[i+1] - temps[i])
            error = (avg_errors[i+1] / (temps[i+1] - temps[i]))**2 + (avg_errors[i] / (temps[i+1] - temps[i]))**2
            error = max(0,error)
            error = np.sqrt(error)
        elif i == len(temps) - 1:  # Backward difference for the last point
            derivative = (avg_predictions[i] - avg_predictions[i-1]) / (temps[i] - temps[i-1])
            error = (avg_errors[i] / (temps[i] - temps[i-1]))**2 + (avg_errors[i-1] / (temps[i] - temps[i-1]))**2
            error = max(0,error)
            error = np.sqrt(error)
        else:  # Central difference for interior points
            derivative = (avg_predictions[i+1] - avg_predictions[i-1]) / (temps[i+1] - temps[i-1])
            error = (avg_errors[i+1] / (temps[i+1] - temps[i-1]))**2 + (avg_errors[i-1] / (temps[i+1] - temps[i-1]))**2
            error = max(0,error)
            error = np.sqrt(error)

        derivatives_of_AvgP.append(derivative)
        errors_in_derivatives.append(error)

    return np.array(derivatives_of_AvgP), np.array(errors_in_derivatives)

# Calculating the energy
def calculate_energy(configuration, J=1):
    energy = 0
    L = configuration.shape[0]
    for i in range(L):
        for j in range(L):
            S = configuration[i, j]
            neighbors = configuration[(i+1) % L, j] + configuration[i, (j+1) % L] + \
                        configuration[(i-1) % L, j] + configuration[i, (j-1) % L]
            energy += -J * S * neighbors
    return energy / (2 * L * L)  # Normalize by the number of spins

# Calculating the binder cumulant
def calculate_binder_cumulant(P, Energies, beta):
    P = np.array(P)
    
    avg_P2 = np.average(P**2)
    avg_P4 = np.average(P**4)
    if avg_P2 == 0:
        raise ValueError("Average of P^2 is zero, leading to division by zero in binder cumulant calculation.")
    binder_cumulant = 1 - (avg_P4 / (3 * avg_P2**2))
    return binder_cumulant


notebook_dir = os.getcwd()
directory = f"/project/ratti/Ahmed/Ising2D_Sim/{L}/Ising2D_MC_Metropolis_CUDA_Sz{L}x{L}_Me2000"
subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

Avg_Predictions = []
Var_Predictions = []
binder_cumulants = []
derivatives_of_AvgP = []
errors_in_derivatives = []

# The independent testing datasets are the outcomes of real MCMC simulations (2000 configuration files per temperature),
# stored in folders, where each folder has a name of a temperature value  

# Loop through subdirectories for temperature calculations
for subdir in subdirectories:
    temp = float(subdir)
    beta = 1.0 / temp  # Inverse temperature

    if temp < 2.27:
        label = 'Ferromagnetic'
    elif temp > 2.27:
        label = 'Paramagnetic'
    else:
        label = ''
    
    
    # Format the temp value to three decimal places and create the file name
    filename = f"{temp:.3f}.txt"
    # Define the folder name
    folder_name = "All_Predictions"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Full path to the text file inside the folder
    all_pred_file_path = os.path.join(folder_name, filename)
    all_predictions = []
    
    subdir_path = os.path.join(directory, subdir)
    file_names = os.listdir(subdir_path)
    
    num_of_files = len(file_names)
    Predictions_at_T = []
    Energies_at_T = []
    
    for file_name in file_names:
        file_path = os.path.join(subdir_path, file_name)
        new_data = np.load(file_path)
        
        # Calculate energy
        energy = calculate_energy(new_data)
        Energies_at_T.append(energy)
        
        new_data_reshaped = new_data.reshape(new_data.shape + (1,))  # Add channel dimension
        new_data_reshaped = np.expand_dims(new_data_reshaped, axis=0)  # Add batch dimension
        
        predictions = model.predict(new_data_reshaped)
        Predictions_at_T.append(predictions[0][1])  # Probability for Ferromagnetic class
        predicted_label = 'Ferromagnetic' if predictions[0][1] > 0.5 else 'Paramagnetic'
        
        ordered_pair = (file_name,predictions[0][1])
        all_predictions.append(ordered_pair)
        
        gc.collect()
    
    
    with open(all_pred_file_path, 'w') as f:
        for pair in all_predictions:
            f.write(f"{pair[0]}:\t\t{pair[1]:.10f}\n")

    
    avg_pred = np.average(Predictions_at_T)
    pseudo_sus = beta * L**2 * ( np.average(np.array(Predictions_at_T)**2) - (np.average(Predictions_at_T))**2)
    pseudo_sus = max(0, pseudo_sus)
    std_err = (np.average(np.array(Predictions_at_T)**2) - (np.average(Predictions_at_T))**2)
    std_err = max(0,std_err)
    std_err = np.sqrt(std_err)
    std_err = std_err/ np.sqrt(num_of_files)
    Avg_Predictions.append((temp, avg_pred, std_err))
    Var_Predictions.append((temp, pseudo_sus))
    
    try:
        binder_cumulant = calculate_binder_cumulant(Predictions_at_T, Energies_at_T, beta)
        binder_cumulants.append((temp, binder_cumulant))
        print(f"Temperature: {temp}, Binder Cumulant: {binder_cumulant}")
    except Exception as e:
        print(f"Error calculating Binder Cumulant for temperature {temp}: {e}")
        binder_cumulants.append((temp, None))
    
    

    
# Save the binder cumulants to a text file
df_BC = pd.DataFrame(binder_cumulants, columns=['Temperature', 'Binder_Cumulant'])
df_BC.sort_values(by='Temperature', inplace=True)
df_BC.to_csv(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_BC_vs_T.txt', sep='\t', index=False, header=False)


Metro_file_path = os.path.join(notebook_dir, f"Ising2D_MC_Metropolis_CUDA_Sz{L}x{L}_Me{saved}.txt")
# Reading the Metro simulation results file
df_Metro = pd.read_csv(Metro_file_path, sep="\t", header=None, names=['Temperature', 'Average Magnetization per Spin'], usecols=[0, 1])
df_Metro.sort_values('Temperature', inplace=True)


# Save the binder cumulants to a text file
df_BC = pd.DataFrame(binder_cumulants, columns=['Temperature', 'Binder_Cumulant'])
df_BC.sort_values(by='Temperature', inplace=True)
df_BC.to_csv(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_BC_vs_T.txt', sep='\t', index=False, header=False)
#################################################################################################################

df_Avg_Predictions = pd.DataFrame(Avg_Predictions, columns=['Temperature', 'Average Prediction per Configuration', 'Error in Prediction'])
df_Avg_Predictions.sort_values('Temperature', inplace=True)
#################################################################################################################
# Get temperatures, average predictions, and errors
temps = df_Avg_Predictions['Temperature'].values
avg_predictions = df_Avg_Predictions['Average Prediction per Configuration'].values
avg_errors = df_Avg_Predictions['Error in Prediction'].values
# Calculate first derivative of average prediction with respect to temperature
derivatives_of_AvgP, errors_in_derivatives = calculate_first_derivative(temps, avg_predictions, avg_errors)

# Save the derivatives to a file
df_derivatives = pd.DataFrame({'Temperature': temps, 'd<AvgP>/dT': derivatives_of_AvgP, 'Error in d<AvgP>/dT': errors_in_derivatives})
df_derivatives.to_csv(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_dAvgP_vs_T.txt', sep=' ', index=False, header=False)
######################################################################################################################

df_Metro = pd.read_csv(Metro_file_path, sep="\t", header=None, names=['Temperature', 'Average Magnetization per Spin', 'Error in Magnetization'], usecols=[0, 1, 2])
df_Metro.sort_values('Temperature', inplace=True)

df_merged_with_errors = pd.merge(df_Avg_Predictions, df_Metro, on='Temperature')


plt.figure(figsize=(10, 6))
plt.errorbar(df_merged_with_errors['Temperature'], df_merged_with_errors['Average Prediction per Configuration'], yerr = df_merged_with_errors['Error in Prediction'], fmt='o', linestyle='-', color='red', label='<P> (Average Prediction)')
plt.errorbar(df_merged_with_errors['Temperature'], df_merged_with_errors['Average Magnetization per Spin'], yerr = df_merged_with_errors['Error in Magnetization'], fmt='o', linestyle='-', color='blue', label='<M> (Average Magnetization per Spin)')
####

plt.title('2D Ising classification model - <P> and <M> vs. Temperature')
plt.xlabel('T')
plt.ylabel('<P> and <M>')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_AvgP_and_AvgM_vs_T.png')

# Save the df_Avg_Predictions DataFrame to a text file
output_file_path = os.path.join(notebook_dir, f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_AvgP_vs_T.txt')
df_Avg_Predictions.to_csv(output_file_path, sep=' ', index=False, header=False)

#################################################################################################################

df_Metro = pd.read_csv(Metro_file_path, sep="\t", header=None, names=['Temperature', 'Average Magnetization per Spin', 'Error in Magnetization', 'Magnetic Susceptibility'], usecols=[0, 1, 2, 3])
df_Metro.sort_values('Temperature', inplace=True)

df_Var_Predictions = pd.DataFrame(Var_Predictions, columns=['Temperature', 'Variance in Prediction'])
df_Var_Predictions.sort_values('Temperature', inplace=True)

# Calculate max_var
max_var = df_Var_Predictions['Variance in Prediction'].max()
# Calculate max_chi
max_chi = df_Metro['Magnetic Susceptibility'].max()
# Define the scaling factor
scaling_factor = max_var / max_chi

df_merged = pd.merge(df_Var_Predictions, df_Metro, on='Temperature')

plt.figure(figsize=(10, 6))
plt.plot(df_merged['Temperature'], df_merged['Variance in Prediction'], marker='o', linestyle='-', color='red', label=r'$\sigma^2$ (Variance in Prediction)')
plt.plot(df_merged['Temperature'], scaling_factor * df_merged['Magnetic Susceptibility'], marker='o', linestyle='-', color='blue', label=r'$\chi$  (Magnetic Susceptibility)')

# Add text to the plot regarding the scaling factor
plt.figtext(0.3, 0.01, f"Magnetic Susceptibility is scaled with a factor {scaling_factor:.2f} for better visualization", fontsize=9, color='red', ha='left')

plt.title(r'2D Ising classification model - $\sigma^2$ and $\chi$ vs. Temperature')
plt.xlabel('T')
plt.ylabel(r'$\sigma^2$ , $\chi$')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_Chi_vs_T.png')

# Save the df_Var_Predictions DataFrame to a text file
output_file_path = os.path.join(notebook_dir, f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_sigma_square_vs_T.txt')
df_Var_Predictions.to_csv(output_file_path, sep=' ', index=False, header=False)

#################################################################################################
# Plot the first derivative with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(df_derivatives['Temperature'], df_derivatives['d<AvgP>/dT'], yerr=df_derivatives['Error in d<AvgP>/dT'], fmt='o', linestyle='-', color='green', label=r'd<AvgP>/dT')
plt.title(r'2D Ising classification model - First derivative of <P> with respect to T')
plt.xlabel('T')
plt.ylabel(r'd<AvgP>/dT')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'Ising2D_Sz{L}_Fi{num_of_total_files}_Tr{num_of_tr}_Te{num_of_ind}_TempRangeB{T_b}A{T_A}_dAvgP_vs_T.png')

