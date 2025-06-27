import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
r = 15  # Communication range
a = 100  # Field size
n = 500  # Number of agents
max_iter = 1000  # PDMM iterations
rho = 0.5  # Penalty parameter (using c instead of rho)

# Generate agent positions
field_range = np.array([[0, a], [0, a]])
sensor_positions = np.array([
    np.random.uniform(field_range[0, 0], field_range[0, 1], n),
    np.random.uniform(field_range[1, 0], field_range[1, 1], n)
])

# Calculate adjacency matrix
def node_connect(ag1, ag2, r,flag):
    return np.sqrt((ag1[0]-ag2[0])**2 + (ag1[1]-ag2[1])**2) <= r

def calculate_adjacency_matrix(positions, r):
    num_agents = positions.shape[1]
    adj_mat = np.zeros((num_agents, num_agents), dtype=int)
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:  # No self-connections
                if i<j: flag = 1
                else: flag = -1
                adj_mat[i,j] =flag * node_connect(positions[:,i], positions[:,j], r,flag)
    return adj_mat

adjacency_matrix = calculate_adjacency_matrix(sensor_positions, r)

def pdmm_median(sensor_values, adjacency_matrix):
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

    for it in range(max_iter):
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

        for i in range(n):
            for j in neighbors[i]:
                z[j][i] = duals[i][j]

    return errors, x_new

# Test the median consensus algorithm
sensor_values = np.random.randn(n) * 10 + 50
errors, values = pdmm_median(sensor_values, adjacency_matrix)
plt.figure(figsize=(10, 6))
plt.semilogy(range(len(errors)), errors, label='No Outliers')


sensor_values_outliers = sensor_values.copy()
ids = np.random.choice(n, 10, replace=False)
sensor_values_outliers[ids] = np.random.randn(10)*5 + 100
errors_outlier, values_outliers = pdmm_median(sensor_values_outliers, adjacency_matrix)
plt.semilogy(range(len(errors_outlier)), errors_outlier)

plt.xlabel('Iteration')
plt.ylabel('Relative Error (log scale)')
plt.title(f'PDMM Median Consensus Convergence (ρ = {rho})')
plt.legend()
plt.grid(True)
plt.show()

print(f"True median: {np.median(sensor_values_outliers):.4f}")
print(f"Final consensus value: {np.mean(values_outliers):.4f}")
print(f"Min agent value: {np.min(values_outliers):.4f}, Max agent value: {np.max(values_outliers):.4f}")
print(f"Final relative error: {errors_outlier[-1]:.4e}")
