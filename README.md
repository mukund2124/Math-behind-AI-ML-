# Answer to the first part
The most appropriate metric to compare distances between any two different paths on a 
constrained 2D grid is the Manhattan distance or the Taxicab distance. This metric is better 
suited than the Euclidean distance because the paths are constrained to horizontal and vertical 
movements, and the Euclidean distance would underestimate the actual distance traveled. 

The Manhattan distance between two points (x1, y1) and (x2, y2) is defined as |x1 - x2| + |y1 - y2|, where |x| represents the absolute value of x. This metric measures the sum of the 
absolute differences between the respective coordinates, effectively counting the number of 
horizontal and vertical moves required to reach the destination from the starting point. 

Other distance metrics that could be considered include: 
• Euclidean distance: √((x1 - x2)^2 + (y1 - y2)^2) 
• Chebyshev distance: max(|x1 - x2|, |y1 - y2|) 
• Minkowski distance: (|x1 - x2|^p + |y1 - y2|^p)^(1/p) 

However, the Manhattan distance is the most appropriate for constrained paths on a 2D grid 
because it accurately reflects the actual distance traveled along the grid lines.It is preferred 
over other metrics, such as the Euclidean distance, because it accurately reflects the actual 
distance traveled on the constrained grid, while the Euclidean distance would underestimate 
the distance by allowing diagonal movements, which are not permitted on the grid. 

Other distance metrics, such as the Chebyshev distance or the Minkowski distance, can also 
be considered, but they may not accurately represent the distance traveled on the constrained 
grid, or they may be computationally more complex than the Manhattan distance. 

# Word Embeddings Visualization 
This project demonstrates how to generate word embeddings, compute their similarity using cosine 
similarity, and visualize them in 3D space using Python and Plotly. We use pre-trained GloVe 
embeddings to represent words numerically and apply PCA for dimensionality reduction. 
## Table of Contents - [Introduction](#introduction) - [Requirements](#requirements) - [Installation](#installation) - [Usage](#usage) 
- [Example](#example) - [License](#license) 
## Introduction 
In this project, we explore word embeddings and their similarities in a high-dimensional space. We 
use the following steps: 
1. Generate a corpus of words. 
2. Convert these words into embeddings using pre-trained GloVe vectors. 
3. Compute the cosine similarity between these embeddings. 
4. Reduce the dimensionality of embeddings using PCA for 3D visualization. 
5. Plot the embeddings in 3D space using Plotly. 
## Requirements - Python 3.6 or higher - `numpy` - `gensim` - `scikit-learn` - `pandas` - `plotly` 
## Installation 
To install the required packages, you can use `pip`: 
```sh 
pip install numpy gensim scikit-learn pandas plotly 
Steps to run the project- 
1.Clone the repository- 
git clone https://github.com/yourusername/word-embeddings-visualization.git 
cd word-embeddings-visualization 
2.Run the script- 
python word_embeddings_visualization.py 
An example script to generate and visualize word embeddings: 
import numpy as np 
import gensim.downloader as api 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.decomposition import PCA 
import plotly.express as px 
import pandas as pd 
# Step 1: Generate a corpus of words 
corpus = ["happiness", "joy", "success", "achievement", "sadness", "sorrow", "failure", "defeat"] 
# Step 2: Load the pre-trained GloVe embeddings 
glove_vectors = api.load("glove-wiki-gigaword-50")  # 50-dimensional GloVe embeddings 
# Get embeddings for the words in the corpus 
embeddings = {word: glove_vectors[word] for word in corpus} 
# Step 3: Convert embeddings dictionary to a matrix 
embedding_matrix = np.array([embeddings[word] for word in corpus]) 
# Compute cosine similarity matrix 
cosine_sim_matrix = cosine_similarity(embedding_matrix) 
# Step 4: Dimensionality reduction using PCA 
pca = PCA(n_components=3) 
reduced_embeddings = pca.fit_transform(embedding_matrix) 
# Step 5: Create a DataFrame for Plotly 
df = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z']) 
df['word'] = corpus 
# Create 3D scatter plot 
fig = px.scatter_3d(df, x='x', y='y', z='z', text='word') 
# Add labels to the plot 
fig.update_traces(marker=dict(size=5), selector=dict(mode='markers+text')) 
# Show plot 
fig.show() 
LICENSE- 
This project is licensed under the MIT License. See the LICENSE file for details. 
To make your project complete, ensure that you include the `word_embeddings_visualization.py` 
script and a `LICENSE` file in your repository. Replace `https://github.com/yourusername/word
embeddings-visualization.git` with the actual URL of your GitHub repository. 
