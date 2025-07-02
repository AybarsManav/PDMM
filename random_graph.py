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
    n = len(A)
    visited = set()

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if abs(A[i][j]) == 1:  # edge exists
                adj[i].append(j)

    def dfs(u):
        visited.add(u)
        for v in adj[u]:
            if v not in visited:
                dfs(v)

    dfs(0)

    return len(visited) == n


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

def plot_graph(sensor_positions, adjacency_matrix, sensor_values=None):
    """
    Visualize the sensor network as a graph.
    sensor_positions: numpy array of shape (2, num_agents)
    adjacency_matrix: numpy array of shape (num_agents, num_agents)
    sensor_values: optional numpy array of shape (num_agents,) for heatmap coloring
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
    # Draw nodes with heatmap coloring if sensor_values provided
    if sensor_values is not None:
        scatter = plt.scatter(
            sensor_positions[0, :], sensor_positions[1, :],
            c=sensor_values, cmap='coolwarm', s=100, zorder=5, edgecolors='k'
        )
        plt.colorbar(scatter, label='Sensor Value')
    else:
        plt.scatter(sensor_positions[0, :], sensor_positions[1, :], c='r', s=30, zorder=5)
    plt.title("Sensor Network Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()