import numpy as np

def pdmm_median(sensor_values, adjacency_matrix, rho, max_transmissions=3e4, tolerance=1e-12, transmission_loss_rate=0.0):
    true_median = np.median(sensor_values)  # Changed to median for error calculation
    n = len(sensor_values)
    neighbors = []
    degrees = []
    for i in range(n):
        nbrs = list(np.where(adjacency_matrix[i] != 0)[0])
        neighbors.append(nbrs)
        degrees.append(len(nbrs))

    x_old = sensor_values.copy()
    duals = [dict() for _ in range(n)]
    z = [dict() for _ in range(n)] 
    for i in range(n):
        for j in neighbors[i]:
            duals[i][j] = 0.0
            z[i][j] = 0

    initial_error = np.linalg.norm(x_old - true_median)
    errors = []
    x_new = np.zeros(n)
    n_transmissions = 0
    current_error = initial_error
    while(max_transmissions > n_transmissions):
        if current_error < tolerance:
            break
        for i in range(n):
            sum_neighbors = sum(z[i][j]*adjacency_matrix[i][j] for j in neighbors[i])
            v_i = -sum_neighbors / (rho * degrees[i]) if degrees[i] > 0 else 0
            
            threshold = 1.0 / (rho * degrees[i]) if degrees[i] > 0 else float('inf')
            diff = v_i - sensor_values[i]
            
            if diff > threshold:
                x_new[i] = v_i - threshold
            elif diff < -threshold:
                x_new[i] = v_i + threshold
            else:
                x_new[i] = sensor_values[i]
            
            for j in neighbors[i]:
                duals[i][j] = z[i][j] + 2*rho*adjacency_matrix[i][j]*x_new[i]
        
        current_error = np.linalg.norm(x_new - true_median)/initial_error
        errors.append(current_error)
        n_transmissions += 1  

        for i in range(n):
            for j in neighbors[i]:
                z[j][i] = 1/2*z[j][i] + 1/2*duals[i][j]

    return errors, x_new