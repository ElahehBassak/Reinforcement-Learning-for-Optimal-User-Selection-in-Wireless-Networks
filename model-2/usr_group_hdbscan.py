#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:32:04 2024

@author: Sara
"""

import numpy as np
import torch
import hdbscan


def corr(H):
    H_tensor = torch.tensor(H, dtype=torch.complex128)
    norms = torch.norm(H_tensor, dim=0)
    
    # Calculate the dot products
    corr_matrix = torch.abs(torch.matmul(H_tensor.conj().T, H_tensor))
    
    # Normalize by the norms
    norm_matrix = torch.outer(norms, norms)
    C = corr_matrix / norm_matrix
    
    # Set diagonal elements to 1 (correlation of a vector with itself)
    C.fill_diagonal_(1)
    
    return C.numpy()

def cluster_with_hdbscan(H, min_cluster_size=2):
     
    """
    #Cluster users using HDBSCAN based on their pairwise correlation.

    Parameters:
    - H (numpy.ndarray): Channel matrix used to calculate the distance matrix. We set the distance matrix to equal the correlation matrix for grouping uncorrelated users together.
    - min_cluster_size (int): The minimum size of clusters to form. 

    Returns:
    - numpy.ndarray: Array of cluster labels for each user. Noise points are labeled as -1.
    """
    #compute Distance Matrix
    dist_matrix = corr(H) #set distance matrix to correlation matrix for grouping uncorrelated users together
    np.fill_diagonal(dist_matrix, 0)
    
    # Initialize HDBSCAN with the distance matrix
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')

    # Fit the model
    clusterer.fit(dist_matrix)

    # Get the cluster labels
    labels = clusterer.labels_

    return labels
    