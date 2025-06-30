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
    # x_new = np.zeros(n)
    x_new = sensor_values.copy()  # Initialize x_new with sensor values
    n_transmissions = 0
    current_error = initial_error
    while(True):

        for i in range(n):
            sum_neighbors = sum(z[i][j]*adjacency_matrix[i][j] for j in neighbors[i])        # Sum of A_ij^T z_i|j
            v_i = -sum_neighbors / (rho * degrees[i]) if degrees[i] > 0 else 0   # - Sum of A_ij^T z_i|j / (rho * d_i)
            
            threshold = 1.0 / (rho * degrees[i]) if degrees[i] > 0 else float('inf')  # 1 / (rho * d_i)
            diff = v_i - sensor_values[i]       # v_i - a_i
            
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

            for j in neighbors[i]:
                z[j][i] = 1/2*z[j][i] + 1/2*duals[i][j] 

            if current_error < tolerance:
                print(f"Error below tolerance ({current_error}) reached.")
                return errors, x_new
            
            if n_transmissions >= max_transmissions:
                print(f"Maximum number of transmissions ({max_transmissions}) reached.")
                return errors, x_new


def pdmm_medianl3(sensor_values, adjacency_matrix, rho, max_transmissions=3e4, tolerance=1e-12, transmission_loss_rate=0.0):
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
    x_new = sensor_values.copy()  # Initialize x_new with sensor values
    n_transmissions = 0
    current_error = initial_error
    while(max_transmissions > n_transmissions):
        if current_error < tolerance:
            break
        for i in range(n):
            sum_neighbors = sum(z[i][j]*adjacency_matrix[i][j] for j in neighbors[i])
            v_i = -sum_neighbors

            a1 = 3
            a2 = -3
            b1 = (rho * degrees[i] - 6 * sensor_values[i])
            b2 = (rho * degrees[i] + 6 * sensor_values[i])
            c1 = -v_i + 3*sensor_values[i]**2 + 1 
            c2 = -v_i - 3*sensor_values[i]**2 - 1
            x_sol_case1 = (-b1 + np.sqrt(b1**2 - 4*a1*c1)) / (2*a1)
            x_sol_case2 = (-b2 + np.sqrt(b2**2 - 4*a2*c2)) / (2*a2)

            if x_sol_case1 > sensor_values[i]:
                x_new[i] = x_sol_case1
            elif x_sol_case2 < sensor_values[i]:
                x_new[i] = x_sol_case2
            else:
                x_new[i] = sensor_values[i]
            
            for j in neighbors[i]:
                duals[i][j] = z[i][j] + 2*rho*adjacency_matrix[i][j]*x_new[i]
        
            current_error = np.linalg.norm(x_new - true_median)/initial_error
            errors.append(current_error)
            n_transmissions += 1
        
            if n_transmissions % 100 == 0:
                print(f"Transmissions: {n_transmissions}, Current Error: {current_error:.6f}")   

        for i in range(n):
            for j in neighbors[i]:
                z[j][i] = duals[i][j]

    return errors, x_new