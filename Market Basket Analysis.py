import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.offline import iplot
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

def zhang_metric(rule):
    association_rule_support = rule['support'].copy()
    association_rule_antecedent = rule['antecedent support'].copy()
    association_rule_consequent = rule['consequent support'].copy()
    x = association_rule_support - (association_rule_antecedent * association_rule_consequent)
    y = np.max((association_rule_support * (1 - association_rule_antecedent).values, association_rule_antecedent * (association_rule_consequent - association_rule_support).values), axis = 0)
    return x / y

groceries_dataframe = pd.read_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\Groceries data.csv')
basket_dataframe = pd.read_csv('C:\\Users\\apoor\\Desktop\\Code\\IT ACADEMIC CODE\\Design Project - 5th Semester\\Basket.csv')

print(len(groceries_dataframe['itemDescription'].unique()))
print(len(groceries_dataframe['Member_number'].unique()))

most_bought_products = groceries_dataframe['itemDescription'].value_counts()

print(most_bought_products.max())
print(most_bought_products.min())

figure = px.bar(data_frame = most_bought_products.head(25), title = 'Most Bought Product', color = most_bought_products.head(25))
figure.update_layout(title_x = 0.5)
figure.show()

user_id = groceries_dataframe['Member_number'].unique()
items = [list(groceries_dataframe.loc[groceries_dataframe['Member_number'] == id, 'itemDescription']) for id in user_id]
print(items[0])

TE = TransactionEncoder()
item_transform = TE.fit(items).transform(items)
item_matrix = pd.DataFrame(item_transform, columns = TE.columns_)
print(item_matrix.head())

apriori_function = apriori(item_matrix, min_support = 0.01, use_colnames = True, max_len = 2).sort_values(by = 'support', ascending = False)
print(apriori_function)

association_rules_apriori = association_rules(apriori_function, metric = 'confidence', min_threshold = 0)
print(association_rules_apriori)

sns.set_context('talk')
sns.relplot(x = 'antecedent support', y = 'consequent support', data = association_rules_apriori, aspect = 1.5, height = 6, size = 'lift', hue = 'confidence')
plt.title('Antecedant Support V/S Consequent Support', fontsize = 15, y = 1.05)
plt.xlabel('Antecedant Support', fontsize = 12)
plt.ylabel('Consequent Support', fontsize = 12)
plt.show()

rule_zhang_metric = zhang_metric(association_rules_apriori)
association_rules_apriori['zhang'] = rule_zhang_metric
print(association_rules_apriori.head())

whole_milk_association_rules = association_rules_apriori[association_rules_apriori['antecedents'].apply(lambda x: 'whole milk' in x)]
whole_milk_association_rules.sort_values('confidence', ascending = False)
print(whole_milk_association_rules)

whole_milk_association_rules_support = whole_milk_association_rules['support'] >= whole_milk_association_rules['support'].quantile(q = 0.9)
whole_milk_association_rules_confidence = whole_milk_association_rules['confidence'] >= whole_milk_association_rules['confidence'].quantile(q = 0.8)
whole_milk_association_rules_lift = whole_milk_association_rules['lift'] > 1
whole_milk_association_rules_zhang = whole_milk_association_rules['zhang'] > 0
ideal_whole_milk_association_rules = whole_milk_association_rules[whole_milk_association_rules_support & whole_milk_association_rules_confidence & whole_milk_association_rules_lift & whole_milk_association_rules_zhang]
print(ideal_whole_milk_association_rules)

association_rules_apriori_visualisation = association_rules_apriori.copy(deep = True)
association_rules_apriori_visualisation_support = association_rules_apriori_visualisation['support'] >= association_rules_apriori_visualisation['support'].quantile(q = 0.8)
association_rules_apriori_visualisation_confidence = association_rules_apriori_visualisation['confidence'] >= association_rules_apriori_visualisation['confidence'].quantile(q = 0.8)
association_rules_apriori_visualisation_lift = association_rules_apriori_visualisation['lift'] > 1
association_rules_apriori_visualisation_zhang = association_rules_apriori_visualisation['zhang'] > 0
ideal_association_rules_apriori_visualisation = association_rules_apriori_visualisation[association_rules_apriori_visualisation_support & association_rules_apriori_visualisation_confidence & association_rules_apriori_visualisation_lift & association_rules_apriori_visualisation_zhang]
print(ideal_association_rules_apriori_visualisation.head(10))

ideal_association_rules_apriori_visualisation['antecedents'] = ideal_association_rules_apriori_visualisation['antecedents'].apply(lambda x: ', '.join(list(x)))
ideal_association_rules_apriori_visualisation['consequents'] = ideal_association_rules_apriori_visualisation['consequents'].apply(lambda x: ', '.join(list(x)))
print(ideal_association_rules_apriori_visualisation)

support_matrix_heatmap = ideal_association_rules_apriori_visualisation.pivot(index = 'antecedents', columns = 'consequents', values = 'support')
plt.style.use('ggplot')
plt.subplots(figsize = (10, 6))
sns_heatmap = sns.heatmap(data = support_matrix_heatmap, annot = True, fmt = '.2f', cmap = 'seismic', cbar = True, annot_kws = {"fontsize": 6})
plt.title('Item\'s Support Matrix', fontsize = 15)
sns_heatmap.set_xlabel('Antecedents', fontsize = 12)
sns_heatmap.set_ylabel('Consequents', fontsize = 12)
plt.xticks(rotation = 60)
plt.tight_layout()
plt.show()

confidence_matrix_heatmap = ideal_association_rules_apriori_visualisation.pivot(index = 'antecedents', columns = 'consequents', values = 'confidence')
figure = ff.create_annotated_heatmap(confidence_matrix_heatmap.to_numpy().round(2), x = list(confidence_matrix_heatmap.columns), y = list(confidence_matrix_heatmap.index), font_colors = ['white', 'white', 'white'], colorscale = ['blue', 'green', 'red'])
figure.update_layout(
    template = 'plotly_dark',
    height = 700,
    width = 1500,
    title = 'Confidence Matrix',
    xaxis_title = 'Consequents',
    xaxis_title_standoff = 5,
    yaxis_title = 'Antecedents',
    title_x = 0.5,
    title_y = 0.98,
    font = dict(
        size = 12,
        color = "white"
    )
)
figure.update_traces(showscale = True)
figure.show()

network_A = list(ideal_association_rules_apriori_visualisation['antecedents'].unique())
network_B = list(ideal_association_rules_apriori_visualisation['consequents'].unique())
nodes_list = list(set(network_A + network_B))

graph = nx.Graph()

for i in nodes_list:
    graph.add_node(i)

for i, j in ideal_association_rules_apriori_visualisation.iterrows():
    graph.add_edges_from([(j['antecedents'], j['consequents'])])

pos = nx.spring_layout(graph, k = 0.5, dim = 2, iterations = 400)

for i, j in pos.items():
    graph.nodes[i]['pos'] = j

edge_trace = go.Scatter(
    x = [],
    y = [],
    hoverinfo = 'none',
    mode = 'lines',
    line = dict(
        width = 0.2,
        color = 'black'
    )
)

for edge in graph.edges():
    x0, y0 = graph.nodes[edge[0]]['pos']
    x1, y1 = graph.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(
    x = [],
    y = [],
    mode = 'markers',
    text = [],
    hoverinfo = 'text',
    marker = dict(
        showscale = True,
        size = 15,
        colorscale = 'tealgrn',
        color = [],
        reversescale = True,
        colorbar = dict(
            thickness = 12,
            title = 'Node Connection',
            titleside = 'right',
            xanchor = 'left'
        )
    ),
    line = dict(width = 0.2)
)

for node in graph.nodes():
    x, y = graph.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(graph.adjacency()):
    node_trace['marker']['color'] += tuple([len(adjacencies[1])])
    node_info = str(adjacencies[0]) + ' has {} connections.'.format(str(len(adjacencies[1])))
    node_trace['text'] += tuple([node_info])

figure = go.Figure(
    data = [edge_trace, node_trace],
    layout = go.Layout(
        title = 'Connection of Maximum Purchased Item',
        titlefont = dict(size = 18),
        showlegend=  False,
        hovermode = 'closest', 
        margin = dict(b = 30, l = 5, r = 5, t = 60),
        xaxis = dict(showgrid = False, zeroline = False),
        yaxis = dict(showgrid = False, zeroline = False)
    )
)

figure.update_layout(title_x = 0.5, title_y = 0.95)
iplot(figure)

best_consequent_whole_milk_association_rules_apriori = association_rules_apriori.copy(deep = True)
best_consequent_whole_milk_association_rules_apriori['antecedents'] = best_consequent_whole_milk_association_rules_apriori['antecedents'].apply(lambda x: ', '.join(x))
best_consequent_whole_milk_association_rules_apriori['consequents'] = best_consequent_whole_milk_association_rules_apriori['consequents'].apply(lambda x: ', '.join(x))

figure = go.Figure()

figure.add_trace(go.Scatter(x = best_consequent_whole_milk_association_rules_apriori['support'], y = best_consequent_whole_milk_association_rules_apriori['zhang'], name = 'All Combinations', mode = 'markers'))
figure.add_trace(go.Scatter(x = ideal_whole_milk_association_rules['support'], y = ideal_whole_milk_association_rules['zhang'], name = 'Top 5 Consequents', mode = 'markers'))

annotation_1 = {'x': '0.191380', 'y': '0.181562', 'showarrow': True, 'arrowhead': 2, 'xshift': -2, 'yshift': 3, 'text': 'Other Vegetables', 'textangle': 0, 'font': {'size': 12, 'color': 'green'}}
annotation_2 = {'x': '0.178553', 'y': '0.189591', 'showarrow': True, 'arrowhead': 2, 'xshift': -2, 'yshift': 3, 'text': 'Rolls/Buns', 'textangle': 0, 'font': {'size': 12, 'color': 'green'}}
annotation_3 = {'x': '0.151103', 'y': '0.091184', 'showarrow': True, 'arrowhead': 2, 'xshift': -2, 'yshift': 3, 'text': 'Soda', 'textangle': 0, 'font': {'size': 12, 'color': 'green'}}
annotation_4 = {'x': '0.150590', 'y': '0.256640', 'showarrow': True, 'arrowhead': 2, 'xshift': -2, 'yshift': 3, 'text': 'Yogurt', 'textangle': 0, 'font': {'size': 12, 'color': 'green'}}
annotation_5 = {'x': '0.116470', 'y': '0.148768', 'showarrow': True, 'arrowhead': 2, 'xshift': -2, 'yshift': 3, 'text': 'Tropical Fruits', 'textangle': 0, 'font': {'size': 12, 'color': 'green'}}

figure.update_layout({'annotations': [annotation_1, annotation_2, annotation_3, annotation_4, annotation_5], 'showlegend': True, 'legend': {'x': 0.88, 'y': 0.02, 'bgcolor': 'rgb(1, 150, 147)'}})
figure.update_xaxes(title_text = 'Support', title_font = {'size': 15}, title_standoff = 10)
figure.update_yaxes(title_text = 'Confidence', title_font = {'size': 15}, title_standoff = 10)
figure.update_layout(title = 'Contribution of Top Items', title_x = 0.5, title_y = 0.9)
figure.show()

figure = px.scatter(best_consequent_whole_milk_association_rules_apriori, x = 'support', y = 'confidence', color = 'lift', hover_data = ['antecedents', 'consequents'], title = 'Support V/S Confidence', labels = {'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'})
figure.update_layout(title_x = 0.5, title_y = 0.9)
figure.show()

network_A = list(best_consequent_whole_milk_association_rules_apriori['antecedents'].unique())
network_B = list(best_consequent_whole_milk_association_rules_apriori['consequents'].unique())
nodes_list = list(set(network_A + network_B))

graph = nx.Graph()

for i in nodes_list:
    graph.add_node(i)

for i, j in best_consequent_whole_milk_association_rules_apriori.iterrows():
    graph.add_edges_from([(j['antecedents'], j['consequents'])])

pos = nx.spring_layout(graph, k = 0.5, dim = 2, iterations = 400)

for i, j in pos.items():
    graph.nodes[i]['pos'] = j

edge_trace = go.Scatter(
    x = [],
    y = [],
    hoverinfo = 'none',
    mode = 'lines',
    line = dict(
        width = 0.2,
        color = 'black'
    )
)

for edge in graph.edges():
    x0, y0 = graph.nodes[edge[0]]['pos']
    x1, y1 = graph.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(
    x = [],
    y = [],
    text = [],
    mode = 'markers',
    hoverinfo = 'text',
    marker = dict(
        showscale = True,
        size = 15,
        colorscale = 'picnic',
        color = [],
        reversescale = False,
        colorbar = dict(
            thickness = 12,
            title = 'Node Connection',
            titleside = 'right',
            xanchor = 'left'
        )
    ),
    line = dict(width = 0.2)
)

for node in graph.nodes():
    x, y = graph.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(graph.adjacency()):
    node_trace['marker']['color'] += tuple([len(adjacencies[1])])
    node_info = str(adjacencies[0]) + ' has {} connections.'.format(str(len(adjacencies[1])))
    node_trace['text'] += tuple([node_info])

figure = go.Figure(
    data = [edge_trace, node_trace],
    layout = go.Layout(
        title = 'All Item Network',
        titlefont = dict(size = 18),
        showlegend=  False,
        hovermode = 'closest', 
        margin = dict(b = 30, l = 5, r = 5, t = 60),
        xaxis = dict(showgrid = False, zeroline = False),
        yaxis = dict(showgrid = False, zeroline = False)
    )
)

figure.update_layout(title_x = 0.5, title_y = 0.95)
iplot(figure)

frequency_basket_items = basket_dataframe.apply(pd.value_counts).transpose().sum().sort_values(ascending = False)
print(frequency_basket_items.head(10))

rainbow_color = plt.cm.rainbow(np.linspace(0, 1, 40))
frequency_basket_items.head(50).plot.bar(color = rainbow_color, figsize = (10, 7))
plt.title('Frequency of Most Purchased Items')
plt.xticks(rotation = 90)
plt.tight_layout()
plt.grid()
plt.show()

basket_dataframe.fillna('N/A', inplace = True)
frequency_basket_items_values_list = basket_dataframe.values.tolist()
print(frequency_basket_items_values_list[0])

for i in range(len(frequency_basket_items_values_list)):
    frequency_basket_items_values_list[i] = [x for x in frequency_basket_items_values_list[i] if not x == 'N/A']

print(frequency_basket_items_values_list[0:10])

TE = TransactionEncoder()
basket_transform = TE.fit(frequency_basket_items_values_list).transform(frequency_basket_items_values_list)
basket_matrix = pd.DataFrame(basket_transform, columns = TE.columns_)
print(basket_matrix.head())

basket_apriori_function = apriori(basket_matrix, min_support = 0.01, use_colnames = True, max_len = 2).sort_values(by = 'support', ascending = False)
print(basket_apriori_function)

basket_association_rules_apriori = association_rules(basket_apriori_function, metric = 'confidence', min_threshold = 0)
print(basket_association_rules_apriori)

sns.set_context('talk')
sns.relplot(x = 'antecedent support', y = 'consequent support', data = basket_association_rules_apriori, aspect = 1.5, height = 6, size = 'lift', hue = 'confidence')
plt.title('Antecedant Support V/S Consequent Support', fontsize = 15, y = 1.05)
plt.xlabel('Antecedant Support', fontsize = 12)
plt.ylabel('Consequent Support', fontsize = 12)
plt.show()

basket_rule_zhang_metric = zhang_metric(basket_association_rules_apriori)
basket_association_rules_apriori['zhang'] = basket_rule_zhang_metric
basket_association_rules_apriori['antecedents'] = basket_association_rules_apriori['antecedents'].apply(lambda x: ', '.join(list(x)))
basket_association_rules_apriori['consequents'] = basket_association_rules_apriori['consequents'].apply(lambda x: ', '.join(list(x)))
print(basket_association_rules_apriori.head())

basket_support_matrix_heatmap = basket_association_rules_apriori.pivot(index = 'antecedents', columns = 'consequents', values = 'support')
plt.style.use('ggplot')
plt.subplots(figsize = (10, 6))
basket_sns_heatmap = sns.heatmap(data = basket_support_matrix_heatmap, annot = True, fmt = '.2f', cmap = 'seismic', cbar = True, annot_kws = {"fontsize": 6})
plt.title("Item's Support Matrix", fontsize = 15)
basket_sns_heatmap.set_xlabel('Antecedents', fontsize = 12)
basket_sns_heatmap.set_ylabel('Consequents', fontsize = 12)
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

basket_confidence_matrix_heatmap = basket_association_rules_apriori.pivot(index = 'antecedents', columns = 'consequents', values = 'confidence')
figure = ff.create_annotated_heatmap(basket_confidence_matrix_heatmap.to_numpy().round(2), x = list(basket_confidence_matrix_heatmap.columns), y = list(basket_confidence_matrix_heatmap.index), font_colors = ['white', 'white', 'white'], colorscale = ['blue', 'green', 'red'])
figure.update_layout(
    template = 'plotly_dark',
    height = 700,
    width = 1500,
    title = 'Confidence Matrix',
    xaxis_title = 'Consequents',
    xaxis_title_standoff = 5,
    yaxis_title = 'Antecedents',
    title_x = 0.5,
    title_y = 0.98,
    font = dict(
        size = 12,
        color = "white"
    )
)
figure.update_traces(showscale = True)
figure.show()

plt.figure(figsize = (10, 7))
scatter_plot = sns.scatterplot(x = 'support', y = 'confidence', data = basket_association_rules_apriori)
plt.title('Support V/S Confidence', fontsize = 15)
scatter_plot.set_xlabel('Support')
scatter_plot.set_ylabel('Confidence')
plt.margins(x = 0.1, y = 0.1)
plt.show()

basket_network = basket_association_rules_apriori[['antecedents', 'consequents']]

basket_network_graph = nx.from_pandas_edgelist(
    basket_association_rules_apriori,
    source = 'antecedents',
    target = 'consequents',
    create_using = nx.DiGraph()
)

basket_in_degree_centrality = nx.in_degree_centrality(basket_network_graph)
basket_network_dataframe  = pd.DataFrame(list(basket_in_degree_centrality.items()), columns = ['items', 'centrality']).dropna(axis = 0)
node_position = nx.kamada_kawai_layout(basket_network_dataframe)

sizes = [x[1] * 100 for x in basket_network_graph.degree()]
print(basket_network_graph.degree())

nx.draw_networkx(
    basket_network_graph,
    with_labels = True,
    node_size = sizes,
    linewidths = 0.5,
    width = 0.2,
    alpha = 0.8,
    arrowsize = 0.5
)

plt.margins(x = 0.1, y = 0.1)
plt.axis('off')
plt.show()
