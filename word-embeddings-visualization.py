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
