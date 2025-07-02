
import numpy as np
import matplotlib.pyplot as plt


def gossip_step(sensor_values, adjacency_matrix, max_ticks, tolerance=1e-12, transmission_loss=0.0):
    true_avg = np.mean(sensor_values) 
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
        if (np.random.rand() > transmission_loss):
            sensor_values[node] = avg_value
        if (np.random.rand() > transmission_loss):
            sensor_values[neighbor] = avg_value
        
        current_error = np.linalg.norm(sensor_values - true_avg)
        relative_error = current_error / initial_error if initial_error > 0 else 0
        errors.append(relative_error)
        if relative_error < tolerance:
            print(f"Error below tolerance ({relative_error}) reached at transmission number {ticks}.")
            break
        
    return sensor_values, errors, abs_time



