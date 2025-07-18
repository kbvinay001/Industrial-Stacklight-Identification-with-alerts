from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from datetime import datetime

def stacklamp_decision_tree(image_data):
    # State mapping from YOLO model
    state_map = {0: 'Red', 1: 'Orange', 2: 'Green', 3: 'Person'}
    machinery_map = {0: 'stopped', 1: 'faulty', 2: 'operational'}

    X = []  
    y_machinery = []  
    y_alert = [] 

    for detected_classes in image_data:
        current_state = 0  
        person_detected = 0  

        for cls in detected_classes:
            if cls in [0, 1, 2]:  
                current_state = cls
            elif cls == 3:  
                person_detected = 1

        machinery_status = current_state

        alert = 0
        if current_state == 2 and person_detected == 1:  n
            alert = 1
        elif current_state == 1:  
            alert = 1

        X.append([current_state, person_detected])
        y_machinery.append(machinery_status)
        y_alert.append(alert)

    X = np.array(X)
    y_machinery = np.array(y_machinery)
    y_alert = np.array(y_alert)

    dt_machinery = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_machinery.fit(X, y_machinery)

    dt_alert = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_alert.fit(X, y_alert)

    data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(len(X))],
        'current_state': [state_map[x[0]] for x in X],
        'person_detected': [bool(x[1]) for x in X],
        'machinery_status': [machinery_map[y] for y in y_machinery],
        'alert': y_alert
    }
    df = pd.DataFrame(data)

    df.to_csv('stacklamp_decision_tree.csv', index=False)
    print("Decision tree results saved to stacklamp_decision_tree.csv")

    return dt_machinery, dt_alert, df

if __name__ == "__main__":
   
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
