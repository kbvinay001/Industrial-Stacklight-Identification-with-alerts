from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from datetime import datetime

# Simulate stacklamp decision tree logic
def stacklamp_decision_tree(image_data):
    # State mapping from YOLO model
    state_map = {0: 'Red', 1: 'Orange', 2: 'Green', 3: 'Person'}
    machinery_map = {0: 'stopped', 1: 'faulty', 2: 'operational'}

    # Prepare data for decision tree
    # Features: current_state (0=Red, 1=Orange, 2=Green), person_detected (0=False, 1=True)
    # Targets: machinery_status (0=stopped, 1=faulty, 2=operational), alert (0=no alert, 1=alert)
    X = []  # Features: [current_state, person_detected]
    y_machinery = []  # Target: machinery_status
    y_alert = []  # Target: alert

    # Process each image's detected classes
    for detected_classes in image_data:
        current_state = 0  # Default: Red
        person_detected = 0  # Default: No person

        # Determine current_state and person_detected from detected classes
        for cls in detected_classes:
            if cls in [0, 1, 2]:  # Stacklamp classes
                current_state = cls
            elif cls == 3:  # Person class
                person_detected = 1

        # Infer machinery status (same as current_state)
        machinery_status = current_state

        # Alert logic
        alert = 0
        if current_state == 2 and person_detected == 1:  # Green + Person
            alert = 1
        elif current_state == 1:  # Orange
            alert = 1

        # Append to dataset
        X.append([current_state, person_detected])
        y_machinery.append(machinery_status)
        y_alert.append(alert)

    # Convert to numpy arrays
    X = np.array(X)
    y_machinery = np.array(y_machinery)
    y_alert = np.array(y_alert)

    # Train decision tree for machinery status
    dt_machinery = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_machinery.fit(X, y_machinery)

    # Train decision tree for alert
    dt_alert = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_alert.fit(X, y_alert)

    # Create output DataFrame
    data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(len(X))],
        'current_state': [state_map[x[0]] for x in X],
        'person_detected': [bool(x[1]) for x in X],
        'machinery_status': [machinery_map[y] for y in y_machinery],
        'alert': y_alert
    }
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('stacklamp_decision_tree.csv', index=False)
    print("Decision tree results saved to stacklamp_decision_tree.csv")

    return dt_machinery, dt_alert, df

# Example usage with simulated detected classes (mimicking YOLO output)
if __name__ == "__main__":
    # Simulated detected classes for each image (class IDs: 0=Red, 1=Orange, 2=Green, 3=Person)
    image_data = [
        [0],         # Red
        [1],         # Orange
        [2],         # Green
        [2, 3],      # Green + Person
        [0, 3],      # Red + Person
        [1, 3],      # Orange + Person
    ]
    dt_machinery, dt_alert, df = stacklamp_decision_tree(image_data)
    print("\nDecision Tree Results:")
    print(df)