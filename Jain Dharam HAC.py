# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 06:40:29 2026

@author: rahi1
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 1. Import the Excel file and specific sheet
file_path = 'D:/LLM Model/Jain Dharam AI/Thematic Analysis Questions.xlsx' # Replace with your actual file path
sheet_name = 'Final_Tablev3'

try:
    df_long = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Data successfully loaded from sheet: {sheet_name}")
except ValueError as e:
    print(f"Error: Could not find sheet '{sheet_name}' in the file '{file_path}'.")
    print(e)


# Display the first few rows of the DataFrame
print(df_long.head())

# --- 2. Prepare Data ---
# 2. Select the relevant columns for clustering
# The data used for clustering should only contain numerical values.
# Prepare Columns List
Value_Col_list = [ i[0] + j for i, j in df_long.loc[:, ['Theme', 'Values']].values]
df = pd.DataFrame({'Samples': ['Aarti', 'Chalisa' , 'Puja']})

for i,j in zip(Value_Col_list, df_long.loc[:, ['Aarti', 'Chalisa' , 'Puja']].values):
    print(j)
    df[i] = j

X = df[Value_Col_list].values
# You can also select the columns by name:
# X = df.loc[:, ['d', 'e', 'f']].values

# (Optional) 3. Standardize the data
# Clustering algorithms that use distance metrics often perform better with scaled data.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- 4. Visualize Results (using scipy for the dendrogram) ---

# Generate the linkage matrix for plotting the dendrogram
# This uses the same linkage method used in the clustering step ('ward' in this example)
linked_matrix = linkage(X, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked_matrix,
           orientation='top',
           distance_sort='descending',
           labels = ['Aarti', 'Chalisa' , 'Puja'],
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Archana Similarity')
plt.ylabel('Distance')
plt.show()