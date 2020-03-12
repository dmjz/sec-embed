import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wordVectorData = (
    {
        'name': 'w2v',
        'dict_fname': 'word_vector_w2v.json',
    },
    {
        'name': 'glove',
        'dict_fname': 'word_vector_glove.json',
    },
)

print('Load data')
with open('dictionary.json', 'r') as f:
        vocabulary = json.load(f)
for d in wordVectorData:
    with open(d['dict_fname'], 'r') as f:
        d['dict'] = json.load(f)

# Filter w2v vectors to use same words as glove vectors
wordVectorData[0]['dict'] = {
    w: wordVectorData[0]['dict'][w] for w in wordVectorData[1]['dict']
}

# for d in wordVectorData:
#     X = list(d['dict'].values())
#     words = list(d['dict'].keys())

#     print('Run K-means')
#     d['kmeans_scores'] = []
#     for K in range(1, 21):
#         kmeans = KMeans(n_clusters=K, random_state=314159)
#         kmeans.fit(X)
#         X_pred = kmeans.fit_predict(X)
#         d['kmeans_scores'].append(kmeans.score(X))
#         print(f'Score (K = { K }): { d["kmeans_scores"][-1] }')

#     print('Save DataFrame of words, clusters, and frequencies')
#     # Note: last computed kmeans will have n_clusters=20, as desired
#     labels = kmeans.labels_
#     centroids = kmeans.cluster_centers_
#     word_kmeans_value = {'word': words, 'cluster': labels}
#     word_kmeans_valuedf = pd.DataFrame(word_kmeans_value)
#     word_kmeans_valuedf['frequency_rank'] = [vocabulary[word] for word in word_kmeans_valuedf['word']]
#     # word_kmeans_valuedf.to_csv('topic_lists.csv')

#     print('Save plot of first five clusters, projected to 2 dimensions')
#     # PCA -> 2 dimensions
#     pca = PCA(n_components=2)
#     pca.fit(X)
#     pca_X = pca.transform(X)
#     pca_X_df = pd.DataFrame(pca_X)
#     pca_X_df.columns = ['PC1', 'PC2']
#     pca_X_df.index = words
#     # Graph of cluster
#     for num_clusters in (3, 5, 7):
#         fig = plt.figure(figsize = (7,7))
#         for cluster in range(num_clusters):
#             index_cluster = (X_pred == cluster)
#             plt.scatter(
#                 pca_X[index_cluster, 0], 
#                 pca_X[index_cluster, 1], 
#                 alpha = 0.5,
#             )
#         fig.savefig(f'pca_cluster_plot_{ num_clusters }_{ d["name"] }.png', dpi=fig.dpi)
#         plt.clf()

# # Plot comparing clustering scores over K
# fig = plt.figure(figsize = (7,7))
# x_ticks = list(range(1, 21))
# for d in wordVectorData:
#     plt.plot(x_ticks, [s/100000 for s in d['kmeans_scores']], label=d['name'])
# plt.xticks([0, 5, 10, 15, 20])
# plt.legend()
# plt.xlabel('K')
# plt.ylabel('Score (100K units)')
# fig.savefig('kmeans_scores_plot.png', dpi=fig.dpi)
# plt.clf()

# Plot PCA clusters for glove vs word2vec on subplots of same figure
pca_plot_data = []
for d in wordVectorData:
    X = list(d['dict'].values())
    words = list(d['dict'].keys())
    kmeans = KMeans(n_clusters=20, random_state=314159)
    X_pred = kmeans.fit_predict(X)
    pca = PCA(n_components=2)
    pca.fit(X)
    pca_X = pca.transform(X)
    pca_X_df = pd.DataFrame(pca_X)
    pca_X_df.columns = ['PC1', 'PC2']
    pca_X_df.index = words
    pca_plot_data.append({
        'name': d['name'],
        'X_pred': X_pred.copy(),
        'pca_X': pca_X.copy(),
    })
    
fig = plt.figure(figsize = (7,7))
fig, (ax1, ax2) = plt.subplots(1, 2)
for ax, pca_data in zip((ax1, ax2), pca_plot_data):
    X_pred = pca_data['X_pred']
    pca_X = pca_data['pca_X']
    for cluster in range(3):
        index_cluster = (X_pred == cluster)
        ax.scatter(
            pca_X[index_cluster, 0], 
            pca_X[index_cluster, 1], 
            alpha = 0.5,
        )
    title = 'Word2Vec' if pca_data['name'] == 'w2v' else 'GloVe'
    ax.set_title(title)
fig.savefig('pca_cluster_comparison_plot.png', dpi=fig.dpi)

