import numpy as np

def pdmm_sync(sensor_values, adjacency_matrix, rho, max_transmissions=3e4, tolerance=1e-12):
    true_avg = np.mean(sensor_values)
    n = len(sensor_values)
    neighbors = []
    degrees = []

    for i in range(n):
        nbrs = list(np.where(adjacency_matrix[i] != 0)[0])
        neighbors.append(nbrs)
        degrees.append(len(nbrs))

    x_old = sensor_values.copy()
    duals = [dict() for _ in range(n)]      # y_i|j
    z = [dict() for _ in range(n)]          # z_i|j
    for i in range(n):
        for j in neighbors[i]:
            duals[i][j] = 0.0
            z[i][j] = 0#x_old[j]

    initial_error = np.linalg.norm(x_old - true_avg)
    errors = []
    x_new = np.zeros(n)
    n_transmissions = 0
    current_error = initial_error
    while(max_transmissions > n_transmissions):
            if current_error < tolerance:
                break

            for i in range(n):
                # NODE UPDATES
                sum_neighbors = sum(z[i][j]*adjacency_matrix[i][j] for j in neighbors[i])           # Sum of A_ij^T z_i|j
                x_new[i] = (sensor_values[i] - sum_neighbors) / (1 + rho * degrees[i])              # x_i update
                for j in neighbors[i]:                                                              # y_i|j update:
                    duals[i][j] = z[i][j] + 2*rho*adjacency_matrix[i][j]*x_new[i]                   # y_i|j = z_i|j + 2c * A_ij^T x_i
                
                # Update error for the current transmission
                current_error = np.linalg.norm(x_new - true_avg)/initial_error                          # Why are we dividing by initial error?
                errors.append(current_error)
                
                # AUXILARY UPDATES (TRANSMISSION)
                n_transmissions += 1  # Transmission is defined as a node updating all of its neighbors
                for j in neighbors[i]:
                    z[j][i] = duals[i][j]   # Swap z_j|i with y_i|j
                
    return errors, x_new, n_transmissions

def pdmm_async(sensor_values, adjacency_matrix, rho, max_transmissions=3e4, tolerance=1e-12):
    true_avg = np.mean(sensor_values)
    n = len(sensor_values)
    neighbors = []
    degrees = []

    for i in range(n):
        nbrs = list(np.where(adjacency_matrix[i] != 0)[0])
        neighbors.append(nbrs)
        degrees.append(len(nbrs))

    x_old = sensor_values.copy()
    duals = [dict() for _ in range(n)]      # y_i|j
    z = [dict() for _ in range(n)]          # z_i|j
    for i in range(n):
        for j in neighbors[i]:
            duals[i][j] = 0.0
            z[i][j] = 0#x_old[j]

    initial_error = np.linalg.norm(x_old - true_avg)
    errors = []
    x_new = np.zeros(n)
    n_transmissions = 0
    current_error = initial_error
    while(max_transmissions > n_transmissions):
            if current_error < tolerance:
                break
            
            # Randomly select a subset of the nodes to update
            update_mask = np.random.randint(0, 2, size=n)
            subset = [i for i in range(n) if update_mask[i] == 1]
            if not subset:
                continue

            for i in subset:
                # NODE UPDATES
                sum_neighbors = sum(z[i][j]*adjacency_matrix[i][j] for j in neighbors[i])           # Sum of A_ij^T z_i|j
                x_new[i] = (sensor_values[i] - sum_neighbors) / (1 + rho * degrees[i])              # x_i update
                for j in neighbors[i]:                                                              # y_i|j update:
                    duals[i][j] = z[i][j] + 2*rho*adjacency_matrix[i][j]*x_new[i]                   # y_i|j = z_i|j + 2c * A_ij^T x_i
                
                # Update error for the current transmission
                current_error = np.linalg.norm(x_new - true_avg)/initial_error                          # Why are we dividing by initial error?
                errors.append(current_error)
                
                # AUXILARY UPDATES (TRANSMISSION)
                n_transmissions += 1  # Transmission is defined as a node updating all of its neighbors
                for j in neighbors[i]:
                    z[j][i] = duals[i][j]   # Swap z_j|i with y_i|j
                
    return errors, x_new, n_transmissions


