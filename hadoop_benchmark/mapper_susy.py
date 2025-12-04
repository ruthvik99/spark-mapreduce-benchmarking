#!/usr/bin/env python3
"""
Mapper for SUSY Dataset - Trains both Logistic Regression and Decision Tree.
SUSY format: first column is label (0/1), rest are features, NO HEADER.
Compatible with Python 3.5+ and old pandas versions.
"""

import sys
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    sys.stderr.write("=== SUSY MAPPER STARTING ===\n")
    
    try:
        # Read entire chunk from stdin
        input_data = sys.stdin.read()
        sys.stderr.write("Read {0} bytes\n".format(len(input_data)))
        
        if not input_data.strip():
            sys.stderr.write("Empty input\n")
            return

        # Parse CSV chunk into DataFrame (NO HEADER for SUSY)
        df = pd.read_csv(StringIO(input_data), header=None)
        sys.stderr.write("Loaded {0} rows, {1} columns\n".format(len(df), df.shape[1]))

        # SUSY: FIRST column is label (0 or 1), rest are features
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        sys.stderr.write("Initial: X shape={0}, y shape={1}\n".format(X.shape, y.shape))

        # Convert to numeric and drop any NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Drop rows with NaN (use isnull() for old pandas compatibility)
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        sys.stderr.write("After cleaning: X shape={0}, y shape={1}\n".format(X.shape, y.shape))

        # Need at least 10 samples
        if len(y) < 10:
            sys.stderr.write("Not enough samples: {0}\n".format(len(y)))
            return

        # If chunk is too large (>50k rows), subsample for memory efficiency
        MAX_ROWS = 50000
        if len(y) > MAX_ROWS:
            sys.stderr.write("Subsampling from {0} to {1} rows\n".format(len(y), MAX_ROWS))
            sample_idx = np.random.choice(len(y), MAX_ROWS, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]
            sys.stderr.write("After subsampling: X shape={0}\n".format(X.shape))

        # Train/test split (local to this mapper)
        sys.stderr.write("Splitting data...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )
        sys.stderr.write("Train: {0}, Test: {1}\n".format(len(X_train), len(X_test)))

        # Two models: Logistic Regression and Decision Tree
        models = [
            ("LogisticRegression",
             LogisticRegression(C=1.0, solver="liblinear", max_iter=100)),
            ("DecisionTree",
             DecisionTreeClassifier(max_depth=5))
        ]

        for name, clf in models:
            try:
                sys.stderr.write("Training {0}...\n".format(name))
                start_time = time.time()

                # Train + predict
                clf.fit(X_train, y_train)
                sys.stderr.write("{0} trained\n".format(name))
                
                predictions = clf.predict(X_test)
                sys.stderr.write("{0} predicted\n".format(name))

                exec_time = time.time() - start_time

                # Basic accuracy
                correct = int((predictions == y_test).sum())
                total = int(len(y_test))

                # Confusion matrix to get TP / FP / FN
                cm = confusion_matrix(y_test, predictions, labels=[0, 1])
                
                # tp = sum of diagonal (correct predictions)
                tp = int(np.diag(cm).sum())
                # fp = sum of columns minus diagonal (false positives)
                fp = int((cm.sum(axis=0) - np.diag(cm)).sum())
                # fn = sum of rows minus diagonal (false negatives)
                fn = int((cm.sum(axis=1) - np.diag(cm)).sum())

                # Emit: model_name \t correct,total,tp,fp,fn,exec_time
                output = "{0}\t{1},{2},{3},{4},{5},{6:.6f}".format(
                    name,
                    correct,
                    total,
                    tp,
                    fp,
                    fn,
                    exec_time
                )
                print(output)
                sys.stdout.flush()
                
                sys.stderr.write("{0} SUCCESS: {1}/{2} correct\n".format(name, correct, total))
                
            except Exception as e:
                sys.stderr.write("Model {0} FAILED: {1}\n".format(name, str(e)))
                import traceback
                sys.stderr.write(traceback.format_exc())
                continue

        sys.stderr.write("=== MAPPER COMPLETED ===\n")

    except Exception as e:
        sys.stderr.write("MAPPER ERROR: {0}\n".format(str(e)))
        import traceback
        sys.stderr.write(traceback.format_exc())


if __name__ == "__main__":
    main()