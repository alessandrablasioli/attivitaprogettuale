import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from geopy.distance import distance, geodesic
import math


'''
---------------------------------
Graphs 
---------------------------------
'''
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


'''
---------------------------------
Functions
---------------------------------
'''


def letter_to_binary(letter):
    unicode_value = ord(letter)
    binary_representation = bin(unicode_value)[2:]  # Remove the prefix '0b'
    return binary_representation


def binary_to_float(binary):
    decimal_value = int(binary, 2)
    return float(decimal_value)

# Function to Calculate New Coordinates (Latitude and Longitude) Given Starting Coordinates
def nuove_coordinate(latitudine, longitudine):
    mean = 0  # Mean of the Gaussian Distribution
    std_deviation = 1
    variazione_metri = np.random.normal(mean, std_deviation) * 100 #modified from 100 to 500, 500 no res, try 1000
    metri_per_grado_lat = 111320.0 
    metri_per_grado_long = 111320.0 * np.cos(np.radians(latitudine/1e7)) 
    variazione_gradi = variazione_metri / metri_per_grado_lat
    variazione_gradi_long = variazione_metri / metri_per_grado_long
  
    # Calculate the new coordinates
    nuova_latitudine = round(latitudine + variazione_gradi*1e7)
    nuova_longitudine = round(longitudine + variazione_gradi_long*1e7)

    return nuova_latitudine, nuova_longitudine



def calcola_distanza(lat1, lon1, lat2, lon2):
    raggio_terrestre = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distanza = raggio_terrestre * c

    return distanza



def trova_valore_piu_vicino(cam_data, target_simulation_time):
    cam_data['time_difference'] = np.abs(cam_data['simulationTime'] - target_simulation_time)
    valore_piu_vicino = cam_data.loc[cam_data['time_difference'].idxmin()]

    return valore_piu_vicino['simulationTime']

