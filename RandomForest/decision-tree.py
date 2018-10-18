import graphviz
import json
import datetime
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

path = "/Users/Halcao/Desktop/NB_IoT.csv"
# with open(path) as f:
#     data = json.load(f)["RECORDS"]
with open(path) as f:
    data = pd.read_csv(f)

data['weekday'], data['hour'] = data['bs_time'].apply(lambda x: pd.Series([datetime.datetime(int('20'+str(x)[0:2]), int(str(x)[2:4]), int(str(x)[4:6])).weekday(), str(x)[6:8]]))

features = ['GPS_x', 'GPS_y', 'weekday', 'hour', 'cellid', 'rssi1', 'earfcn']
prediction = ['rsrp1']
X = np.array(data[features])  # Create an array
y = np.array(data[prediction])

# regt = DecisionTreeRegressor(max_depth=4)

# regt = regt.fit(X, y)  # Build a decision tree regressor from the training set (X, y)

# dot_data = tree.export_graphviz(regt, out_file=None)  # Export a decision tree in DOT format

# graph = graphviz.Source(dot_data)

# graph.render("tree")  # Save the source to file
rfr = RandomForestRegressor()
rfr.fit(X, y)
# print(rfr.predict([X[100]]))