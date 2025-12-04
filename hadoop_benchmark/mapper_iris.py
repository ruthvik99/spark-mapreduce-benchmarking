#!/usr/bin/env python3
"""
Mapper for Iris Dataset - Trains both Logistic Regression and Decision Tree.
Compatible with Hadoop Streaming and Python 3.5+.

"""

import sys
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def main():
    try:
        # Read entire chunk from stdin
        input_data = sys.stdin.read()
        if not input_data.strip():
            return

        # Parse CSV chunk into DataFrame
        df = pd.read_csv(StringIO(input_data))

        # Skip tiny chunks
        if len(df) < 5:
            return

        # Iris: last column is label, rest are features
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Encode string labels as integers
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Train/test split (local to this mapper)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

        # Two models: Logistic Regression and Decision Tree
        models = [
            ("LogisticRegression",
             LogisticRegression(C=1.0, solver="liblinear", multi_class="ovr")),
            ("DecisionTree",
             DecisionTreeClassifier(max_depth=5))
        ]

        for name, clf in models:
            try:
                start_time = time.time()

                # Train + predict
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                exec_time = time.time() - start_time

                # Basic accuracy
                correct = int((predictions == y_test).sum())
                total = int(len(y_test))

                # Confusion matrix to get TP / FP / FN (micro style)
                cm = confusion_matrix(y_test, predictions)
                # cm is (num_classes x num_classes)
                # tp = sum of diagonal
                tp = int(np.diag(cm).sum())
                # fp = sum of column minus diagonal
                fp = int((cm.sum(axis=0) - np.diag(cm)).sum())
                # fn = sum of row minus diagonal
                fn = int((cm.sum(axis=1) - np.diag(cm)).sum())

                # Emit: model_name \t correct,total,tp,fp,fn,exec_time
                print(
                    "{0}\t{1},{2},{3},{4},{5},{6:.6f}".format(
                        name,
                        correct,
                        total,
                        tp,
                        fp,
                        fn,
                        exec_time
                    )
                )
            except Exception:
                # Skip model failures for this chunk and continue
                continue

    except Exception:
        # Fail silently on malformed chunks
        pass


if __name__ == "__main__":
    main()