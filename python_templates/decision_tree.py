from sklearn.tree import DecisionTreeRegressor


tree = DecisionTreeRegressor()
tree.fit(x, y)
y_hat = tree.predict(y)
