####################################################################################################
# This test is executed as following: mpirun -np n_procs python scalability_tests/test_opinf_mpi.py
####################################################################################################

import numpy as np

from simulai.metrics import L2Norm
from simulai.regression import OpInf

# Number of samples and number of variables
N_SAMPLES = 10000
N_VARS = 50

# Generate input and output data
data_input = np.random.rand(N_SAMPLES, N_VARS)
data_output = np.random.rand(N_SAMPLES, N_VARS)

# Set the first column of the input data to 1
data_input[:, 0] = 1

# List of batch sizes to test
batch_sizes = [10, 100, 1000]

# Regularization parameters
lambda_linear = 1
lambda_quadratic = 1

# Lists to store the results
D_o_list = []
R_matrix_list = []

# Iterate over the batch sizes
for batch_size in batch_sizes:
    # Create an OpInf object
    model = OpInf(solver="lstsq", parallel="mpi")

    # Set the regularization parameters and fit the model
    model.set(lambda_linear=lambda_linear, lambda_quadratic=lambda_quadratic)
    model.fit(
        input_data=data_input,
        target_data=data_output,
        batch_size=batch_size,
        continuing=False,
    )

    # Store the results
    D_o_list.append(model.D_o)
    R_matrix_list.append(model.R_matrix)

# Get the reference results
ref_D_o = D_o_list.pop(0)
ref_R_matrix = R_matrix_list.pop(0)

# Create an L2Norm object
l2_norm = L2Norm()

# Iterate over the results and compare them to the reference results
for ii, (d_o, r_matrix) in enumerate(zip(D_o_list, R_matrix_list)):
    # Check if the matrices are equal
    if not np.all(np.isclose(ref_D_o, d_o).astype(int)):
        print(
            f"The case with batch_size {batch_sizes[ii]} is divergent for the matrix D_o"
        )
    if not np.all(np.isclose(ref_R_matrix, r_matrix).astype(int)):
        print(
            f"The case with batch_size {batch_sizes[ii]} is divergent for the matrix R_matrix"
        )

    # Calculate the error and deviation
    error_d_o = l2_norm(data=d_o, reference_data=ref_D_o, relative_norm=True)
    error_r_matrix = l2_norm(
        data=r_matrix, reference_data=ref_R_matrix, relative_norm=True
    )
    maximum_deviation_d_o = np.abs(d_o - ref_D_o).max()
    maximum_deviation_r_matrix = np.abs(r_matrix - ref_R_matrix).max()

    # Print the results
    print(f"Maximum D_o error: {100 * error_d_o} %.")
    print(f"Maximum R_matrix error: {100 * error_r_matrix} %.")
    print(f"Maximum D_o deviation: {maximum_deviation_d_o}.")
    print(f"Maximum R_matrix deviation: {maximum_deviation_r_matrix}.")
