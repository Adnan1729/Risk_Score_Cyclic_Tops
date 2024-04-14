from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.patches import Patch, Rectangle
import os

#ML Parts
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv(r'C:\Users\AMahmud1\OneDrive - Network Rail\Documents\Theory\CT Score\CT_OWW_2100_Peak_Amplitudes\OWW-2100-134404-222214-2021-05-20.csv')
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['18_top_right'] = pd.to_numeric(df['18_top_right'], errors='coerce')
df.dropna(subset=['Mileage', '18_top_right'], inplace=True)

# Create a mask where '18_top_right' exceeds 3
mask= (df['18_top_right'] >= 2) 
df['group'] = (mask != mask.shift()).cumsum()
filtered_groups = df[mask].groupby('group').filter(lambda x: len(x) >= 3)
filtered_groups['regime_id'] = filtered_groups['group'].astype('category').cat.codes + 1
df = df.join(filtered_groups[['regime_id']], how='outer')
df['regime_id'].fillna(0, inplace=True)
df = df[df['regime_id'] != 0]

# Calculate features
features = df.groupby('regime_id').agg(
    number_of_peaks=('18_top_right', 'count'),
    variability_of_peak_size=('18_top_right', 'std'),
    total_y_value=('18_top_right', 'sum')
)

# Convert total_y_value to 'energy'
features['energy'] = features['total_y_value'] * 0.8
features.drop(columns='total_y_value', inplace=True)

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features[['number_of_peaks', 'variability_of_peak_size', 'energy']])
features_scaled = pd.DataFrame(features_scaled, columns=['number_of_peaks', 'variability_of_peak_size', 'energy'])

# Define and apply weights
weights = np.array([0.05, 0.05, 0.9])
weighted_features = features_scaled * weights

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=0)
features['cluster'] = kmeans.fit_predict(weighted_features)

# Calculate the mean values of the original features within each cluster
cluster_means = features.groupby('cluster').mean()

# Define cluster names
cluster_names = {
    0: 'High Pre-Risk',
    1: 'Medium Pre-Risk',
    2: 'Low Pre-Risk',
    3: 'Moderate',
    4: 'Minimal'
}
features['cluster_name'] = features['cluster'].map(cluster_names)

df_final = pd.merge(df, features, left_on='regime_id', right_index=True, how='left')
output_columns = ['Mileage', '18_top_right', 'regime_id', 'number_of_peaks', 'variability_of_peak_size', 'energy', 'cluster', 'cluster_name']
df_output = df_final[output_columns]
df_output.to_csv(r'C:\Users\AMahmud1\Downloads\output104.csv', index=False)

# Visualisation
# Create a color mapping for each cluster
cluster_colors = {
    'High Pre-Risk': 'red',
    'Medium Pre-Risk': 'blue',
    'Low Pre-Risk': 'green',
    'Moderate': 'purple',
    'Minimal': 'orange'
}

# Apply the color mapping to 'cluster_name' in df_output to create a 'color' column
df_output['color'] = df_output['cluster_name'].map(cluster_colors)

# Ensure that there's a default color for any data points not in any cluster, if necessary
df_output['color'].fillna('grey', inplace=True)  # or choose another neutral color as default

df_all = pd.read_csv(r'C:\Users\AMahmud1\OneDrive - Network Rail\Documents\Theory\CT Score\CT_OWW_2100_Peak_Amplitudes\OWW-2100-134404-222214-2021-05-20.csv')
df_all['Mileage'] = pd.to_numeric(df_all['Mileage'], errors='coerce')
df_all['18_top_right'] = pd.to_numeric(df_all['18_top_right'], errors='coerce')
df_all.dropna(subset=['Mileage', '18_top_right'], inplace=True)

# Define the window size for the plot
window_size = .1

# Function to adjust the x-axis format
def adjust_xaxis_tick_format(ax, window_size):
    tick_spacing = window_size / 20
    ax.xaxis.set_major_locator(MaxNLocator(20))  # Limits the number of ticks to 20
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    ax.figure.canvas.draw()

# Function for the interactive scrolling plot
def scroll_x(window_center):
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Calculate window bounds
    window_start = window_center - (window_size / 2)
    window_end = window_center + (window_size / 2)
    
    # Plot all points in df
    mask = (df_all['Mileage'] >= window_start) & (df_all['Mileage'] <= window_end)
    filtered_df = df_all[mask]
    ax.scatter(filtered_df['Mileage'], filtered_df['18_top_right'], color='grey', alpha=0.5, s=30, marker="s")
    ax.vlines(filtered_df['Mileage'], ymin=0, ymax=filtered_df['18_top_right'], colors='black', lw=0.5) 
    
    # Overlay the points from df_output with clusters, with unique colors
    mask_output = (df_output['Mileage'] >= window_start) & (df_output['Mileage'] <= window_end)
    filtered_df_output = df_output[mask_output]
    ax.scatter(filtered_df_output['Mileage'], filtered_df_output['18_top_right'], c=filtered_df_output['color'], s=30, marker="s")
    ax.vlines(filtered_df_output['Mileage'], ymin=0, ymax=filtered_df_output['18_top_right'], colors='black', lw=0.5) 

    legend_patches = [Patch(facecolor=color, label=cluster_name, alpha=0.5) for cluster_name, color in cluster_colors.items()]
    for name, group in filtered_df_output.groupby('regime_id'):
        min_x = group['Mileage'].min()
        max_x = group['Mileage'].max()
        min_y = 0
        max_y = df['18_top_right'].max() + 0.5
        # Use the regime's cluster color for the rectangle
        regime_color = cluster_colors.get(group['cluster_name'].iloc[0], 'grey')
        ax.add_patch(Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, facecolor=regime_color, alpha=0.2, edgecolor='none'))
        

    
    
    # Set plot limits and labels
    ax.set_xlim([window_start, window_end])
    ax.set_ylim(0, max(df['18_top_right'].max() + 0.5, 6))  # Slightly above the highest line for visibility
    ax.set_xlabel('Mileage')
    ax.set_ylabel('18_top_right')
    ax.grid(True)
    adjust_xaxis_tick_format(ax, window_size)
    
    # Horizontal lines for reference
    plt.axhline(y=5, color='red', linestyle='-', linewidth=0.5)
    plt.axhline(y=2, color='orange', linestyle='-', linewidth=0.5)

    # Show legend if within the current window
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    legend = ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=len(legend_patches))
    plt.show()

# Initial center of the window
initial_center = df_all['Mileage'].min() + (window_size / 2)

# Create the slider
slider = FloatSlider(min=df_all['Mileage'].min(), max=df_all['Mileage'].max(), step=0.0001, value=initial_center, description="Scroll X:", readout_format='.4f')

# Display the interactive plot
interactive(scroll_x, window_center=slider)




