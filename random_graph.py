import numpy as np
import matplotlib.pyplot as plt

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
    laplacian_matrix = D - abs(A)

    eigenvals = np.linalg.eigvalsh(laplacian_matrix)
    zero_eigenvalues = np.sum(np.isclose(eigenvals, 0))

    # Graph is connected if there is exactly one zero eigenvalue
    return zero_eigenvalues == 1

def calculate_adjacency_matrix(positions, r):
    num_agents = positions.shape[1]
    adj_mat = np.zeros((num_agents, num_agents), dtype=int)
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:  # No self-connections
                if i<j: flag = 1
                else: flag = -1
                adj_mat[i,j] = flag * node_connect(positions[:,i], positions[:,j], r)
    return adj_mat

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