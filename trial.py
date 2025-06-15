
import numpy as np
import matplotlib.pyplot as plt

#find something to support a usual communication range
r = 15
a = 100
np.random.seed(42)

def node_connect(ag1,ag2, r):
    '''
    Check if two agents are connected based on their positions and communication range.
    ag1, ag2: tuples or lists containing (x, y) coordinates of the agents
    r: communication range
    Returns True if connected, False otherwise.
    '''
    return(np.sqrt(np.sum((ag1[1]-ag2[1])**2 + (ag1[0]-ag2[0])**2)) <= r)

def is_connected(A):
    """
    Check if the graph represented by adjacency matrix A is connected.
    A: numpy array representing the adjacency matrix of the graph
    Returns True if the graph is connected, False otherwise.
    """
    D = np.diag(np.sum(A, axis=1))
    laplacian_matrix = D - A

    eigenvals = np.linalg.eigvalsh(laplacian_matrix)
    zero_eigenvalues = np.sum(np.isclose(eigenvals, 0))

    # Graph is connected if there is exactly one zero eigenvalue
    return zero_eigenvalues == 1

def calculate_adjacency_matrix(sensor_positions, r):
    # agent_locations: [GP_INPUT_DIM, NUM_AGENTS]
    """ 
    Calculate the adjacency matrix for a set of agents based on their positions and communication range.
    sensor_positions: numpy array of shape (2, num_agents) where each column is the position of an agent
    2 is for x and y coordinates
    r: communication range
    Returns: adjacency matrix as a numpy array of shape (num_agents, num_agents)
    """ 

    num_agents = sensor_positions.shape[1]
    adjacency_matrix = np.zeros((num_agents, num_agents), dtype=int)
    for i in range(num_agents):
        for j in range(num_agents):
            adjacency_matrix[i,j] = node_connect(sensor_positions[:,i],sensor_positions[:,j],r)

    return adjacency_matrix

    
#for now i will put some number of agents

n = 120

x = np.linspace(0, a, 100)
y = np.linspace(0, a, 100)
xx, yy = np.meshgrid(x, y)
field_range = np.array([[0, a], [0, a]])

# generate agent positions [GP_INPUT_DIM, NUM_AGENTS]
sensor_positions = np.array([0.9*np.random.uniform(field_range[0,0], field_range[0,1], n,),
                            0.9*np.random.uniform(field_range[1,0], field_range[1,1], n)])

adjacency_matrix = calculate_adjacency_matrix(sensor_positions, r)
sensor_values = np.random.rand(n)


if not is_connected(adjacency_matrix):
    print("The graph is not connected.")
else:
    print("The graph is connected.")

def plot_graph(sensor_positions, adjacency_matrix):
    """
    Visualize the sensor network as a graph.
    sensor_positions: numpy array of shape (2, num_agents)
    adjacency_matrix: numpy array of shape (num_agents, num_agents)
    """
    num_agents = sensor_positions.shape[1]
    plt.figure(figsize=(8, 8))
    # Draw edges
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            if adjacency_matrix[i, j]:
                x = [sensor_positions[0, i], sensor_positions[0, j]]
                y = [sensor_positions[1, i], sensor_positions[1, j]]
                plt.plot(x, y, 'b-', alpha=0.3)
    # Draw nodes
    plt.scatter(sensor_positions[0, :], sensor_positions[1, :], c='r', s=30, zorder=5)
    plt.title("Sensor Network Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

plot_graph(sensor_positions, adjacency_matrix)


def gossip_step(sensor_values, adjacency_matrix, max_ticks, true_avg):
    n = len(sensor_values)
    initial_error = np.linalg.norm(sensor_values - true_avg)  # For normalization
    
    ticks = 0
    abs_time = 0
    errors = []
    
    neighbors_dict = {}
    for i in range(n):
        neighbors_dict[i] = np.where(adjacency_matrix[i] > 0)[0].tolist()
    
    while ticks < max_ticks:
        intertick = np.random.exponential(1/n)
        abs_time += intertick
        ticks += 1
        node = np.random.randint(n)
        
        if len(neighbors_dict[node]) == 0:
            continue  
        
        neighbor = np.random.choice(neighbors_dict[node])
        
        avg_value = (sensor_values[node] + sensor_values[neighbor]) / 2.0
        sensor_values[node] = avg_value
        sensor_values[neighbor] = avg_value
        
        current_error = np.linalg.norm(sensor_values - true_avg)
        relative_error = current_error / initial_error if initial_error > 0 else 0
        errors.append(relative_error)
        
    return sensor_values, errors, abs_time


values = np.random.randn(n) * 10 + 50
true_avg = np.mean(values)
#values[50] = 200

max_events = 1000
end_values, errors, abs_time = gossip_step(values.copy(), adjacency_matrix, max_events, true_avg)

# Plot results
t = np.linspace(0, abs_time, len(errors))
plt.figure(figsize=(10, 6))
plt.semilogy(t, errors)
plt.xlabel('Absolute Time (seconds)')
plt.ylabel('Relative Error (log scale)')
plt.title('Gossip Algorithm Convergence')
plt.grid(True)
plt.show()

print(f"Initial values: {np.min(values):.2f} to {np.max(values):.2f}")
print(f"Final values: {np.min(end_values):.2f} to {np.max(end_values):.2f}")
print(f"True average: {true_avg:.4f}")
print(f"Events: {max_events}, Time: {abs_time:.2f}s")
print(f"Avg activations per node: {max_events/n:.2f}")


rho =0.5

neighbors = []
degrees = []
for i in range(n):
    nbrs = list(np.where(adjacency_matrix[i] > 0)[0])
    neighbors.append(nbrs)
    degrees.append(len(nbrs))

# Initialize ADMM variables
x_old = values
#z0 = np.array(np.random.randn(2*n) * 10 + 50) #random  init for consensus var
duals = [dict() for _ in range(n)]
z0 = [dict() for _ in range(n)]
for i in range(n):
    for j in neighbors[i]:
        duals[i][j] = 20
        z0[i][j] = float((np.random.rand(1)*10 + 50).item()) #get some random values in z0

initial_error = np.linalg.norm(x_old - true_avg)
errors = []
x_new = values
z = z0
# PDMM iterations
#y = 2(u-z) z is 2m, y is 2m 
for it in range(max_events):
    
    # primal update for each node
    for i in range(n):
        #sum_duals = sum(duals[i][j] for j in neighbors[i])
        #sum_neighbors = sum(x_old[j] for j in neighbors[i])
        #x_new[i] = (sensor_values[i] - sum_duals + rho * sum_neighbors) / (1 + rho * degrees[i])
        x_new[i] = (values[i] - sum(z[i][j]*adjacency_matrix[i][j] for j in neighbors[i])) / (1 + rho * degrees[i])
        for j in neighbors[i]:
            duals[i][j] = z[i][j] + 2*rho*(adjacency_matrix[i][j]*x_new[i])
    for i in range(n):
        for j in neighbors[i]:
            duals[j][i] = duals[i][j]
    for i in range(n):
        for j in neighbors[i]:
            z[j][i] = duals[i][j]
    # relative error
    current_error = np.linalg.norm(x_new - true_avg) / initial_error
    errors.append(current_error)
    print(np.mean(x_new))
    
  
    # dual update for each edge
    #for i in range(n):
    #    for j in neighbors[i]:
            # Update dual variable using new x values
    #        duals[i][j] += z * + 2*rho(adjacency_matrix[i][j]*x_new)
    
    
            

    x_old = x_new

#trying out outliers
values = np.random.randn(n) * 10 + 50
true_avg = np.mean(values)
#values[3] = 400
#values[50] = values [60] = values[89] = values[100] = values[3] 

'''
x_old_outlier = sensor_values.copy()
duals_outlier = [dict() for _ in range(n)]
for i in range(n):
    for j in neighbors[i]:
        duals_outlier[i][j] = 0.0

initial_error_outlier = np.linalg.norm(x_old - true_avg)
errors_outlier = []


for it in range(max_events):
    x_new_outlier = np.zeros(n)
    
    # primal update for each node
    for i in range(n):
        sum_duals_outlier = sum(duals_outlier[i][j] for j in neighbors[i])
        sum_neighbors_outlier = sum(x_old_outlier[j] for j in neighbors[i])
        x_new_outlier[i] = (sensor_values[i] - sum_duals_outlier + rho * sum_neighbors_outlier) / (1 + rho * degrees[i])
    
    # relative error
    current_error_outlier = np.linalg.norm(x_new_outlier - true_avg) / initial_error_outlier
    errors_outlier.append(current_error_outlier)
    
    # dual update for each edge
    for i in range(n):
        for j in neighbors[i]:
            # Update dual variable using new x values
            duals[i][j] += rho * (x_new_outlier[i] - x_new_outlier[j])
    
    x_old_outlier = x_new_outlier
'''
# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogy(range(max_events), errors, label = 'No Outliers')
#plt.semilogy(range(max_events), errors_outlier,label = 'Outliers')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Relative Error (log scale)')
plt.title('Proximal ADMM Convergence for Distributed Averaging')
plt.grid(True)
plt.show()

# Print results
print(f"True average: {true_avg:.4f}")
print(f"Final average across agents: {np.mean(x_new):.4f}")
print(f"Min agent value: {np.min(x_new):.4f}, Max agent value: {np.max(x_new):.4f}")
print(f"Final relative error: {errors[-1]:.4e}")