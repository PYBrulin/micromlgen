> [!WARNING]
> This fork does not follow the structure or versioning of the original project. Changes have been made to the code to fix certain problems and add new features, but it is not guaranteed to be stable or to work as intended from the original project, nor will the original documentation apply to this fork.
>
> The main features added are the generation of pure c-code for models instead of the Cpp-Arduino templates/namespaces; in addition to corrections to the original code.
>
> Please note that as this fork may have diverged from the original project for my own use, I won't be making a pull request to the original project.

# Introducing MicroML

MicroML is an attempt to bring Machine Learning algorithms to microcontrollers.
Please refer to [this blog post](https://eloquentarduino.github.io/2019/11/you-can-run-machine-learning-on-arduino/)
to an introduction to the topic.

## Install

`pip install git+https://github.com/PYBrulin/micromlgen.git`

## Supported classifiers

`micromlgen` can port to plain C many types of classifiers:

- DecisionTree
- RandomForest
- XGBoost
- GaussianNB
- Support Vector Machines (SVC and OneClassSVM)
- Relevant Vector Machines (from `skbayes.rvm_ard_models` package)
- SEFR
- PCA

```python
from micromlgen import port
from sklearn.svm import SVC
from sklearn.datasets import load_iris


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = SVC(kernel='linear').fit(X, y)
    print(port(clf))
```

You may pass a classmap to get readable class names in the ported code

```python
from micromlgen import port
from sklearn.svm import SVC
from sklearn.datasets import load_iris


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = SVC(kernel='linear').fit(X, y)
    print(port(clf, classmap={
        0: 'setosa',
        1: 'virginica',
        2: 'versicolor'
    }))
```

## PCA

It can export a PCA transformer.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from micromlgen import port

if __name__ == '__main__':
    X = load_iris().data
    pca = PCA(n_components=2, whiten=False).fit(X)

    print(port(pca))
```

## SEFR

Read the post about [SEFR](https://eloquentarduino.github.io/2020/07/sefr-a-fast-linear-time-classifier-for-ultra-low-power-devices/).

```shell script
pip install sefr
```

```python
from sefr import SEFR
from micromlgen import port


clf = SEFR()
clf.fit(X, y)
print(port(clf))
```

## DecisionTreeRegressor and RandomForestRegressor

```bash
pip install micromlgen>=1.1.26
```

```python
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from micromlgen import port


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    regr = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5).fit(X, y)
    regr = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=5).fit(X, y)

    with open('RandomForestRegressor.h', 'w') as file:
        file.write(port(regr))
```

```cpp
// Arduino sketch
#include "RandomForestRegressor.h"

Eloquent::ML::Port::RandomForestRegressor regressor;
float X[] = {...};


void setup() {
}

void loop() {
    float y_pred = regressor.predict(X);
}
```
