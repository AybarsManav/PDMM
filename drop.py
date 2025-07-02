import numpy as np
from random_graph import is_connected

# Helper function to drop nodes
def drop_nodes(sensor_values, adjacency_matrix, num_drop=20):
    n = adjacency_matrix.shape[0]
    ids = np.random.choice(n, num_drop, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[ids] = False
    new_sensor_values = sensor_values[mask]
    new_adjacency_matrix = adjacency_matrix[np.ix_(mask, mask)]
    return new_sensor_values, new_adjacency_matrix

# Usage example (before calling PDMM functions):
# sensor_values, adjacency_matrix = drop_nodes(sensor_values, adjacency_matrix, num_drop=20)

def pdmm_sync_drop(sensor_values, adjacency_matrix, rho, max_transmissions=3e4, tolerance=1e-12, transmission_loss_rate=0.0, num_drop=20):
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
    dropped = False
    while(max_transmissions > n_transmissions):
            if current_error < tolerance:
                break
            if n_transmissions > 2000 and not dropped:  # Drop nodes after 2000 transmissions
                dropped = True
                sensor_values, adjacency_matrix = drop_nodes(sensor_values, adjacency_matrix, num_drop)
                is_connected(adjacency_matrix)  # Check if the graph is still connected
                print(f"Nodes dropped: {num_drop}, Remaining nodes: {len(sensor_values)}")
                print(f"Graph is connected after drop: {is_connected(adjacency_matrix)}")
                n = len(sensor_values)
                neighbors = []
                degrees = []
                for i in range(n):
                    nbrs = list(np.where(adjacency_matrix[i] != 0)[0])
                    neighbors.append(nbrs)
                    degrees.append(len(nbrs))
                x_old = sensor_values.copy()
                duals = [dict() for _ in range(n)]
                z = [dict() for _ in range(n)]  # <-- reinitialize z!
                for i in range(n):
                    for j in neighbors[i]:
                        duals[i][j] = 0.0
                        z[i][j] = 0
                x_new = x_old.copy()  
                true_avg = np.mean(sensor_values)  
                initial_error = np.linalg.norm(x_old - true_avg)  
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
                    if np.random.rand() >= transmission_loss_rate: # Update only if transmission loss does not occur
                        z[j][i] = duals[i][j]   # Swap z_j|i with y_i|j
                
    return errors, x_new, n_transmissions