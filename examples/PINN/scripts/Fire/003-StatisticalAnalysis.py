import csv
import numpy as np
from scipy import interpolate

# Load experimental data from CSV file
experimental_data = []
with open('001-Fire_ODEs_Dataset.csv', 'r') as experimental_file:
    reader = csv.reader(experimental_file)
    next(reader)  # Skip the first line
    for line in reader:
        # Add time and experimental data to the list
        time = float(line[0])
        value = float(line[1])
        experimental_data.append((time, value))

# Load simulation data from CSV file
simulation_data = []
with open('002-Fire_PINN_Dataset.csv', 'r') as simulation_file:
    reader = csv.reader(simulation_file)
    next(reader)  # Skip the first line
    for line in reader:
        # Add time and simulation data to the list
        time = float(line[0])
        value = float(line[1])
        simulation_data.append((time, value))

# Separate time and experimental values
experimental_time, experimental_values = zip(*experimental_data)

# Interpolate simulation data
simulation_time, simulation_values = zip(*simulation_data)
interp_simulation_values = interpolate.interp1d(simulation_time, simulation_values)(experimental_time)

# Calculate R2
mean_experimental_values = np.mean(experimental_values)
SSE = np.sum((experimental_values - interp_simulation_values) ** 2)
SST = np.sum((experimental_values - mean_experimental_values) ** 2)
R2 = 1 - (SSE / SST)

print(f"R\u00b2: {R2:.4f}")

