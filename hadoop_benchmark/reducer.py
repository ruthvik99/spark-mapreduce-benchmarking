#!/usr/bin/env python3
"""
Reducer that aggregates:
- Accuracy
- F1-score
- Execution Time (FIXED: uses max time instead of sum)
- Throughput
Python 3.5 compatible (no f-strings).
Handles multiple model name formats.
"""

import sys
import time

# Capture reducer start time for total job time measurement
REDUCER_START_TIME = time.time()

def normalize_model_name(name):
    """
    Normalize model names to consistent format.
    LogisticRegression -> logistic_regression
    DecisionTree -> decision_tree
    """
    name = name.strip().lower()
    # Handle camelCase by inserting underscore before capitals
    result = ""
    for i, char in enumerate(name):
        if i > 0 and char.isupper():
            result += "_"
        result += char.lower()
    
    # Also handle spaces
    result = result.replace(" ", "_")
    
    # Map variations to standard names
    if "logistic" in result:
        return "logistic_regression"
    elif "decision" in result or "tree" in result:
        return "decision_tree"
    else:
        return result

def main():
    # Metrics stored per model
    models = {
        "logistic_regression": {
            "correct": 0, 
            "total": 0, 
            "tp": 0, 
            "fp": 0, 
            "fn": 0, 
            "max_time": 0.0,  # CHANGED: Track max time instead of sum
            "total_time": 0.0,  # Keep sum for reference
            "mappers": 0
        },
        "decision_tree": {
            "correct": 0, 
            "total": 0, 
            "tp": 0, 
            "fp": 0, 
            "fn": 0, 
            "max_time": 0.0,  # CHANGED: Track max time instead of sum
            "total_time": 0.0,  # Keep sum for reference
            "mappers": 0
        }
    }

    line_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        line_count += 1
        
        try:
            # Parse model and metrics
            parts = line.split("\t")
            if len(parts) != 2:
                sys.stderr.write("Line {0}: Expected 2 parts, got {1}\n".format(line_count, len(parts)))
                continue
                
            key = parts[0]
            values = parts[1]
            
            # Normalize model name
            model = normalize_model_name(key)
            sys.stderr.write("Line {0}: Raw key='{1}', Normalized='{2}'\n".format(line_count, key, model))

            value_parts = values.split(",")
            if len(value_parts) != 6:
                sys.stderr.write("Line {0}: Expected 6 values, got {1}\n".format(line_count, len(value_parts)))
                continue

            correct = int(value_parts[0])
            total = int(value_parts[1])
            tp = int(value_parts[2])
            fp = int(value_parts[3])
            fn = int(value_parts[4])
            exec_time = float(value_parts[5])

            if model not in models:
                sys.stderr.write("Line {0}: Unknown model '{1}'\n".format(line_count, model))
                continue

            # Aggregate
            models[model]["correct"] += correct
            models[model]["total"] += total
            models[model]["tp"] += tp
            models[model]["fp"] += fp
            models[model]["fn"] += fn
            models[model]["total_time"] += exec_time
            models[model]["max_time"] = max(models[model]["max_time"], exec_time)  # CHANGED
            models[model]["mappers"] += 1
            
            sys.stderr.write("Line {0}: SUCCESS - Added to {1} (time: {2:.4f}s)\n".format(
                line_count, model, exec_time))

        except Exception as e:
            sys.stderr.write("Line {0}: Exception: {1}\n".format(line_count, str(e)))
            continue

    # Calculate total reducer execution time
    reducer_total_time = time.time() - REDUCER_START_TIME
    
    sys.stderr.write("Processed {0} lines total\n".format(line_count))
    sys.stderr.write("Reducer execution time: {0:.4f}s\n".format(reducer_total_time))

    # ---- Output Block (Python 3.5 compatible) ----

    for model, stats in models.items():
        print("=" * 60)
        print("{0} RESULTS".format(model.upper().replace("_", " ")))
        print("=" * 60)

        total = stats["total"]
        correct = stats["correct"]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        # CHANGED: Use max_time as the execution time
        # This represents the slowest mapper (bottleneck)
        time_to_report = stats["max_time"]
        
        mappers = stats["mappers"]

        if total > 0:
            accuracy = correct / float(total)
            precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
            recall = tp / float(tp + fn) if tp + fn > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            throughput = total / time_to_report if time_to_report > 0 else 0.0
        else:
            accuracy = precision = recall = f1 = throughput = 0.0

        print("Accuracy:\t\t{0:.4f}".format(accuracy))
        print("Precision:\t\t{0:.4f}".format(precision))
        print("Recall:\t\t\t{0:.4f}".format(recall))
        print("F1-score:\t\t{0:.4f}".format(f1))
        print("Correct:\t\t{0}".format(correct))
        print("Total Samples:\t\t{0}".format(total))
        print("Execution Time (s):\t{0:.4f}".format(time_to_report))
        print("Total Time All Mappers:\t{0:.4f}".format(stats["total_time"]))
        print("Avg Time Per Mapper:\t{0:.4f}".format(stats["total_time"] / mappers if mappers > 0 else 0.0))
        print("Throughput (rows/s):\t{0:.2f}".format(throughput))
        print("Mappers Used:\t\t{0}\n".format(mappers))


if __name__ == "__main__":
    main()