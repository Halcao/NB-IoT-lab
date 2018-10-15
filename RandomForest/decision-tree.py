import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor

path = "/Users/Halcao/Desktop/NB_IoT.csv"
# with open(path) as f:
#     data = json.load(f)["RECORDS"]
with open(path) as f:
    data = pd.read_csv(f)

features = ['areacode', 'cellid', 'tx_power', 'cell_id1', 'rssi1']
predict = ['GPS_x', 'GPS_y']
X = np.array(data[features])  # Create an array
y = np.array(data[predict])

# regt = DecisionTreeRegressor(max_depth=4)

# regt = regt.fit(X, y)  # Build a decision tree regressor from the training set (X, y)

# dot_data = tree.export_graphviz(regt, out_file=None)  # Export a decision tree in DOT format

# graph = graphviz.Source(dot_data)

# graph.render("tree")  # Save the source to file
rfr = RandomForestRegressor()
rfr.fit(X, y)
print(rfr.predict([X[100]]))