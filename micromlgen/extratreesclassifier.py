from micromlgen.utils import check_type, jinja


def is_extratrees(clf) -> bool:
    """Test if classifier can be ported"""
    return check_type(clf, "ExtraTreesClassifier")


def port_extratrees(clf, **kwargs) -> str:
    """Port sklearn's ExtraTreesClassifier"""
    # Note: Use RandomForest.jinja as default template for Forest classifiers
    return jinja(
        "randomforest/randomforest.jinja",
        {
            "n_classes": (clf.n_classes_ if isinstance(clf.n_classes_, int) else len(clf.n_classes_)),
            "trees": [
                {
                    "left": clf.tree_.children_left,
                    "right": clf.tree_.children_right,
                    "features": clf.tree_.feature,
                    "thresholds": clf.tree_.threshold,
                    "classes": clf.tree_.value,
                }
                for clf in clf.estimators_
            ],
        },
        {"classname": "ExtraTrees"},
        **kwargs,
    )
