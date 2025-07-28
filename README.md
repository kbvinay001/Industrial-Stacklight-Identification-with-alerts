
# Industrial Safety System with Stacklight & Person Detection 🚦👷

This project uses computer vision to enhance industrial safety by monitoring machinery status through stacklight colors and detecting human presence in proximity to active equipment. It leverages a **YOLOv11** model to identify stacklight colors (simulated with traffic lights) and persons, triggering specific alerts based on the operational context.

-----

## 🚦 What are Industrial Stacklights?

An **industrial stacklight** (or signal tower light) is a common fixture on manufacturing equipment. It provides a quick visual and sometimes audible indication of a machine's status to personnel in the vicinity.

The colors typically have standard meanings:

  * 🟢 **Green:** Normal operation. The machine is active and running as intended.
  * 🟡 **Amber/Orange:** Warning or fault. The machine requires attention, may be low on materials, or has encountered a non-critical error.
  * 🔴 **Red:** Critical failure or emergency stop. The machine is stopped due to a serious issue or a hazardous condition.

This project uses these color-coded signals as the primary input for determining machinery status.

-----

## 🚨 Custom Alert System

This system implements a custom logic to generate real-time alerts based on the detected objects (light color and person). The goal is to create context-aware notifications.

| Detected State | Machinery Status | Alert Message |
| :--- | :--- | :--- |
| 🟢 **Green Light** | Working Normally | *No alert.* |
| 🟢 **Green Light** + 👤 **Person** | Working Normally | `PERSON IS PRESENT NEAR THE WORK MACHINERY, ALERT!!` |
| 🟡 **Orange Light** | Fault Condition | `FAULT IN MACHINERY, NEED A TECHNICIAN!` |
| 🔴 **Red Light** | Halted/Stopped | *No alert is generated.* |

-----

## 🤔 Why YOLOv11 and Not YOLOv8?

While **YOLOv8** is the latest state-of-the-art model from Ultralytics, known for its high efficiency and accuracy, this project was developed using **YOLOv11**.

The choice of a specific model architecture often depends on project requirements, familiarity with the codebase, or the pursuit of specific research goals. YOLOv11 was chosen for this implementation to explore its specific architecture and performance on our custom dataset of traffic lights and persons. While YOLOv8 might offer a more streamlined experience and top-tier performance out-of-the-box, working with different versions like v11 allows for a broader understanding of the evolution and variations within the YOLO family of models.

-----

## 📜 Key Scripts Explained

The core logic of this project is divided into two main scripts: the real-time inference engine and a decision tree model.

### `inference.py`

This is the main execution script. It performs the following steps:

1.  Loads the trained YOLOv11 model (`best.pt`).
2.  Captures input from a video file or live camera feed.
3.  Performs object detection on each frame to identify light colors and people.
4.  Implements the custom `if-else` logic defined in the **Custom Alert System** section to display the machine's status and trigger alerts.

### `decision_tree.py`

This script serves as a formal model of our alerting logic. Instead of just implementing the rules with `if-else` statements, this code uses a **Decision Tree Classifier** from `scikit-learn` to replicate the decision-making process.

**Why is this useful?**

  * **Explicit Modeling:** It explicitly models the rules learned from the input data (detected classes) to predict the output (alerts).
  * **Visualization:** The resulting tree can be visualized to provide a clear, flowchart-like representation of the decision logic, making it easy to understand and verify.
  * **Validation:** It demonstrates that our simple, rule-based alert system can be represented by a classical machine learning model, validating the logic's structure.

-----
