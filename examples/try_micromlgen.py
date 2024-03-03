from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: F401
from xgboost import XGBClassifier  # noqa: F401

from micromlgen import port, port_testset

if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    # regr = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5).fit(X, y)
    # regr = RandomForestRegressor(n_estimators=2, max_depth=10, min_samples_leaf=5).fit(X, y)
    # regr = LogisticRegression(max_iter=100).fit(X, y)
    clf = DecisionTreeClassifier().predict()
    clf = RandomForestClassifier(n_estimators=10)
    clf = XGBClassifier(n_estimators=10)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    with open('examples/classifier.h', 'w') as file:
        file.write(port(clf, classname='XGBClassifier', tmp_file='examples/xgboost.json'))

    print(port_testset(X[:10], y_pred[:10], classname='Iris'))
