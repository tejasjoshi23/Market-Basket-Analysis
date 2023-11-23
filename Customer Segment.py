import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def biplot(no_outlier_dataframe, data_reduction, pca):
    figure, axis = plt.subplots(figsize = (10, 7))
    axis.scatter(x = data_reduction.loc[:, 'Dimension 1'], y = data_reduction.loc[:, 'Dimension 2'], facecolors = 'b', edgecolors = 'b', s = 10)

    feature_vector = pca.components_.T
    arrow_size, text_position = 7.0, 7.0

    for i, j in enumerate(feature_vector):
        axis.arrow(0, 0, arrow_size * j[0], arrow_size * j[1], head_width = 0.2, head_length = 0.2, linewidth = 1, color = 'green')
        axis.text(j[0] * text_position, j[1] * text_position, no_outlier_dataframe.columns[i], fontsize = 15, ha = 'center', va = 'center', color = 'black')

    axis.set_xlabel('Dimension 1', fontsize = 12)
    axis.set_ylabel('Dimension 2', fontsize = 12)
    axis.set_title('PCA with Original Feature Projection', fontsize = 15)

    return axis

def cluster_results(data_reduction, cluster_predict, center, pca_random_indices_dataframe):
    prediction = pd.DataFrame(cluster_predict, columns = ['Cluster'])
    plot_data = pd.concat([prediction, data_reduction], axis = 1)

    figure, axis = plt.subplots(figsize = (10, 6))

    color_map = cm.get_cmap('gist_rainbow')
    
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax = axis, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', label = 'Cluster %i' % (i), color = color_map((i)*1.0/(len(center)-1)))
       
    for i, c in enumerate(center):
        axis.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', alpha = 1, linewidths = 2, marker = 'o', s = 300)
        axis.scatter(x = c[0], y = c[1], marker = '$%d$' % (i), alpha = 1, s = 100)

    axis.scatter(x = pca_random_indices_dataframe[:, 0], y = pca_random_indices_dataframe[:, 1], s = 150, linewidths = 2, color = 'black', marker = 'x')

    axis.set_title('Cluster Learning on PCA - Centroids Marked by Numbers, Sample Data Marked by Black Cross')

    plt.tight_layout()
    plt.show()

def silhouette_coefficient(number_cluster):
    cluster = KMeans(n_clusters = number_cluster, random_state = 0)
    cluster.fit(data_reduction.values)
    cluster_predict = cluster.predict(data_reduction.values)

    cluster_center = cluster.cluster_centers_
    sample_cluster_predict = cluster.predict(pca_random_indices_dataframe)

    score = silhouette_score(data_reduction, cluster_predict)

    print('Silhouette Coefficient for {} clusters: {:.4f}'.format(number_cluster, score))

def Gaussian_Mixture(k):
    global cluster, predict, center, sample_predict

    cluster = GaussianMixture(n_components = k, random_state = 0)
    cluster.fit(data_reduction.values)
    predict = cluster.predict(data_reduction.values)

    center = cluster.means_
    sample_predict = cluster.predict(pca_random_indices_dataframe)

    score = silhouette_score(data_reduction, predict)
    return score

def channel_result(data_reduction, outliers, pca_random_indices_dataframe):
    try:
        full_dataframe = dataframe
    except:
        print('ERROR!! Dataset could not be loaded.')
        return False

    channel_dataframe = pd.DataFrame(full_dataframe['Channel'], columns = ['Channel'])
    channel_dataframe = channel_dataframe.drop(channel_dataframe.index[outliers]).reset_index(drop = True)
    labels = pd.concat([data_reduction, channel_dataframe], axis = 1)

    figure, axis = plt.subplots(figsize = (10, 6))

    color_map = cm.get_cmap('gist_rainbow')
    plot_label = ['Hotel/Restaurant/Cafe', 'Retailers']
    
    for i, cluster in labels.groupby('Channel'):
        cluster.plot(ax = axis, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', color = color_map((i-1)*1.0/2), label = plot_label)
       
    for i, c in enumerate(center):
        axis.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', alpha = 1, linewidths = 2, marker = 'o', s = 300, facecolors = 'none')
        axis.scatter(x = c[0] + 0.25, y = c[1] + 0.3, marker = '$%d$' % (i), alpha = 1, s = 100)

    axis.scatter(x = pca_random_indices_dataframe[:, 0], y = pca_random_indices_dataframe[:, 1], s = 150, linewidths = 2, color = 'black', marker = 'x')

    axis.set_title("PCA-Reduced Data Labeled by 'Channel' - Transformed Sample Data Circled")

    plt.tight_layout()
    plt.show()

dataframe = pd.read_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\customers.csv')
dataframe.drop(['Channel', 'Region'], axis = 1, inplace = True)
print(dataframe.head())

groceries_dataframe = pd.read_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\Groceries data.csv')
print(groceries_dataframe['itemDescription'].unique())

print(dataframe.isnull().sum())
print(dataframe.dtypes)

print(dataframe.describe())

random_indices = [51, 159, 385]
random_indices_dataframe = pd.DataFrame(dataframe.loc[random_indices], columns = dataframe.keys())
print(random_indices_dataframe)

feature_drop = 'Detergents_Paper'

new_dataframe = dataframe.drop(feature_drop, axis = 1)
labels = dataframe[feature_drop]

x_train, x_test, y_train, y_test = train_test_split(new_dataframe, labels, test_size = 0.25, random_state = 0)

tree_regressor = DecisionTreeRegressor(random_state = 0)
tree_regressor.fit(x_train, y_train)

score = tree_regressor.score(x_test, y_test)
print(score)

scatter_matrix_plot = pd.plotting.scatter_matrix(dataframe, figsize = (10, 7), diagonal = 'hist')

for ax in scatter_matrix_plot.flatten():
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.show()

bar_plot = sns.barplot(data = dataframe, palette = 'seismic')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

sns.violinplot(data = dataframe, palette = 'hot')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

sns.heatmap(dataframe.corr(), annot = True)
plt.tight_layout()
plt.show()

log_dataframe = np.log(dataframe)
log_random_indices_dataframe = np.log(random_indices_dataframe)

log_scatter_matrix_plot = pd.plotting.scatter_matrix(log_dataframe, figsize = (10, 7), diagonal = 'hist')

for ax in log_scatter_matrix_plot.flatten():
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.show()

print(log_random_indices_dataframe)

outliers = []

for x in log_dataframe.keys():
    Quarter_1 = np.percentile(log_dataframe[x], 25)
    Quarter_3 = np.percentile(log_dataframe[x], 75)

    step = (Quarter_3 - Quarter_1) * 1.5

    print('Outlier Step: ', step)

    print("Data points considered outlier for the feature '{}': ".format(x))
    feature_outlier = log_dataframe[~((log_dataframe[x] >= Quarter_1 - step) & (log_dataframe[x] <= Quarter_3 + step))]
    print(feature_outlier)

    outliers += feature_outlier.index.tolist()

no_outlier_dataframe = log_dataframe.drop(log_dataframe.index[outliers]).reset_index(drop = True)

print('Number of outliers (including duplicates):', len(outliers))
print('New dataset with removed outliers has {} samples with {} features each.'.format(*no_outlier_dataframe.shape))

pca = PCA(n_components = 6).fit(no_outlier_dataframe)
pca_log = pca.transform(log_random_indices_dataframe)

pca = PCA(n_components = 2).fit(no_outlier_dataframe)
data_reduction = pca.transform(no_outlier_dataframe)
pca_random_indices_dataframe = pca.transform(log_random_indices_dataframe)

data_reduction = pd.DataFrame(data_reduction, columns = ['Dimension 1', 'Dimension 2'])

print(pd.DataFrame(np.round(pca_random_indices_dataframe, 4), columns = ['Dimension1', 'Dimension 2']))

biplot(no_outlier_dataframe, data_reduction, pca)
plt.tight_layout()
plt.show()

for i in range(2, 11):
    silhouette_coefficient(i)

result = pd.DataFrame(columns = ['Silhouette Score'])
result.index.name = 'Number of Clusters'

for i in range(2, 11):
    score = Gaussian_Mixture(i)
    result = pd.concat([result, pd.DataFrame([score], columns = ['Silhouette Score'], index = [i])])

print(result)

cluster = KMeans(n_clusters = 2)
cluster.fit(data_reduction.values)
cluster_predict = cluster.predict(data_reduction.values)
cluster_center = cluster.cluster_centers_
sample_cluster_predict = cluster.predict(pca_random_indices_dataframe)
cluster_results(data_reduction, cluster_predict, cluster_center, pca_random_indices_dataframe)

cluster_1 = GaussianMixture(n_components = 2)
cluster_1.fit(data_reduction.values)
cluster_1_predict = cluster_1.predict(data_reduction.values)
cluster_1_center = cluster_1.means_
sample_cluster_1_predict = cluster_1.predict(pca_random_indices_dataframe)
cluster_results(data_reduction, cluster_1_predict, cluster_1_center, pca_random_indices_dataframe)

log_center = pca.inverse_transform(cluster_1_center)
actual_center = np.exp(log_center)

segments = ['Segment {}'.format(i) for i in range(0, len(cluster_1_center))]

actual_center = pd.DataFrame(np.round(actual_center), columns = dataframe.keys())
actual_center.index = segments
print(actual_center)

plt.figure(figsize = (8, 6))
plt.axes().set_title('Segment 0')
sns_barplot = sns.barplot(x = actual_center.columns.values, y = actual_center.iloc[1].values)
plt.tight_layout()
plt.show()

plt.figure(figsize = (8, 6))
plt.axes().set_title('Segment 1')
sns_barplot = sns.barplot(x = actual_center.columns.values, y = actual_center.iloc[1].values)
plt.tight_layout()
plt.show()

for i, prediction in enumerate(sample_cluster_1_predict):
    print('Sample point ', i, ' to be predicted in Cluster ', prediction)

channel_result(data_reduction, outliers, pca_random_indices_dataframe)