import numpy as np
import matplotlib.pyplot as plt

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

values = np.random.randn(n) * 10 + 50
true_avg = np.mean(values)
#values[50] = 200

max_events = 1000


def run_admm(sensor_values, adjacency_matrix, rho, max_iter=100):
    """Run ADMM with a given rho and return convergence errors."""
    n = len(sensor_values)
    neighbors = []
    degrees = []
    for i in range(n):
        nbrs = list(np.where(adjacency_matrix[i] > 0)[0])
        neighbors.append(nbrs)
        degrees.append(len(nbrs))

    x_old = sensor_values.copy()
    duals = [dict() for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:
            duals[i][j] = 0.0

    initial_error = np.linalg.norm(x_old - true_avg)
    errors = []

    for it in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            sum_duals = sum(duals[i][j] for j in neighbors[i])
            sum_neighbors = sum(x_old[j] for j in neighbors[i])
            x_new[i] = (sensor_values[i] - sum_duals + rho * sum_neighbors) / (1 + rho * degrees[i])

        current_error = np.linalg.norm(x_new - true_avg) / initial_error
        errors.append(current_error)

        for i in range(n):
            for j in neighbors[i]:
                duals[i][j] += rho * (x_new[i] - x_new[j])

        x_old = x_new

    return errors

rho_candidates = [0.01, 0.05, 0.1, 0.5, 1.0]#, 2.0, 5.0, 10.0]
results = {}

for rho in rho_candidates:
    errors = run_admm(sensor_values.copy(), adjacency_matrix, rho, max_iter=100)
    results[rho] = errors

plt.figure(figsize=(10, 6))
for rho, errors in results.items():
    plt.semilogy(range(len(errors)), errors, label=f"ρ = {rho}")

plt.xlabel('Iteration')
plt.ylabel('Relative Error (log scale)')
plt.title('ADMM Convergence for Different ρ Values')
plt.legend()
plt.grid(True)
plt.show()

best_rho = min(results.keys(), key=lambda rho: results[rho][-1])
print(f"Best ρ: {best_rho} (lowest final error: {results[best_rho][-1]:.4e})")