import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from pdmm import *
from pdmm_median import *

def rho_search_pdmm_sync(adjacency_matrix, sensor_values):
    rho_candidates = [0.01, 0.05, 0.1, 0.5, 1.0]
    results = {}
    for rho in rho_candidates:
        errors, values, n_transmissions = pdmm_sync(sensor_values.copy(), adjacency_matrix,rho)
        results[rho] = errors
    best_rho = min(results.keys(), key=lambda rho: results[rho][-1])
    return best_rho, results


def rho_search_pdmm_async(adjacency_matrix, sensor_values):
    rho_candidates = [0.01, 0.05, 0.1, 0.5, 1.0]
    results = {}
    for rho in rho_candidates:
        errors, values, n_transmissions = pdmm_async(sensor_values.copy(), adjacency_matrix,rho)
        results[rho] = errors
    best_rho = min(results.keys(), key=lambda rho: results[rho][-1])
    return best_rho, results


def rho_search_pdmm_median(adjacency_matrix, sensor_values):
    rho_candidates = [0.01, 0.05, 0.1, 0.5, 1.0]
    results = {}
    for rho in rho_candidates:
        errors, x = pdmm_median(sensor_values.copy(), adjacency_matrix,rho)
        results[rho] = errors
    best_rho = min(results.keys(), key=lambda rho: results[rho][-1])
    return best_rho, results