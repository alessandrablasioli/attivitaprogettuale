import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from geopy.distance import distance, geodesic



def plot_clustering(data,clusters):
# Otteniamo i dati come array NumPy 
    data_array = np.array(data)

    # Creiamo un dizionario che associa l'ID ad un colore
    unique_ids = np.unique(data_array[:, 0]).astype(int)
    num_ids = len(unique_ids)
    id_colors = plt.cm.get_cmap('hsv', num_ids)
    id_color_dict = {unique_ids[i]: id_colors(i) for i in range(num_ids)}

    # Creiamo il grafico scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cluster in enumerate(clusters):
        cluster_data = data_array[cluster]
        cluster_ids = cluster_data[:, 0].astype(int)
        cluster_colors = [id_color_dict[cluster_id] for cluster_id in cluster_ids]
        ax.scatter(cluster_data[:, 2], cluster_data[:, 1], c=cluster_colors, alpha=0.5, label=f'Cluster {i+1}')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()


def plot_clustering_with_centroids(data,clusters,centroids):
# Otteniamo i dati come array NumPy 
    data_array = np.array(data)

    # Creiamo un dizionario che associa l'ID ad un colore
    unique_ids = np.unique(data_array[:, 0]).astype(int)
    num_ids = len(unique_ids)
    id_colors = plt.cm.get_cmap('hsv', num_ids)
    id_color_dict = {unique_ids[i]: id_colors(i) for i in range(num_ids)}

    # Creiamo il grafico scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cluster in enumerate(clusters):
        cluster_data = data_array[cluster]
        cluster_ids = cluster_data[:, 0].astype(int)
        cluster_colors = [id_color_dict[cluster_id] for cluster_id in cluster_ids]
        ax.scatter(cluster_data[:, 2], cluster_data[:, 1], c=cluster_colors, alpha=0.5, label=f'Cluster {i+1}')
        ax.scatter(centroids[i][2], centroids[i][1], c='red', marker='x', s=200)  # disegna il centroide come un punto rosso
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

def plot_two(data1, data2, clusters1, clusters2):
    # First set of data
    data_array1 = np.array(data1)

    # Second set of data
    data_array2 = np.array(data2)

    # Create a dictionary that maps cluster index to a color
    num_clusters1 = len(clusters1)
    cluster_colors1 = plt.cm.get_cmap('hsv', num_clusters1)

    # Assign black and brown colors to clusters in the second dataset
    cluster_colors2 = ['black', 'brown']

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot first set of data
    for i, cluster in enumerate(clusters1):
        cluster_data = data_array1[cluster]
        cluster_color = cluster_colors1(i / num_clusters1)  # Assign color based on cluster index
        ax.scatter(cluster_data[:, 2], cluster_data[:, 1], c=cluster_color, alpha=0.5, label=f'Cluster 1-{i+1}')

    # Plot second set of data
    for i, cluster in enumerate(clusters2):
        cluster_data = data_array2[cluster]
        cluster_color = cluster_colors2[i % len(cluster_colors2)]  # Cycle through black and brown colors
        ax.scatter(cluster_data[:, 2], cluster_data[:, 1], c=cluster_color, alpha=0.5, label=f'Cluster 2-{i+1}')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Show the plot
    plt.show()


#plotting dove si utilizza l'event typer per scegliere i colori

def plot_clustering_event(data,clusters):
# Otteniamo i dati come array NumPy 
    data_array = np.array(data)

    # Creiamo un dizionario che associa l'ID ad un colore
    unique_events = np.unique(data_array[:, 4]).astype(int)
    num_events = len(unique_events)
    ev_colors = plt.cm.get_cmap('hsv', num_events)
    ev_color_dict = {unique_events[i]: ev_colors(i) for i in range(num_events)}

    # Creiamo il grafico scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cluster in enumerate(clusters):
        cluster_data = data_array[cluster]
        cluster_ids = cluster_data[:, 4].astype(int)
        cluster_colors = [ev_color_dict[cluster_id] for cluster_id in cluster_ids]
        ax.scatter(cluster_data[:, 2], cluster_data[:, 1], c=cluster_colors, alpha=0.5, label=f'Cluster {i+1}')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()