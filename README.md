
# Industrial Safety System with Stacklight & Person Detection 🚦👷

This project uses computer vision to enhance industrial safety by monitoring machinery status through stacklight colors and detecting human presence in proximity to active equipment. It leverages a custom-trained **YOLO** model within the `ultralytics` framework to identify stacklight colors (simulated with traffic lights) and persons, triggering specific alerts and logging the data.

-----

## 🚀 Project Workflow

This project follows a clear, sequential workflow from setup to execution:

1.  **Environment Setup**: A dedicated virtual environment is created using **Anaconda** to manage all project dependencies and avoid conflicts.
2.  **Model Training (`train.py`)**: A custom YOLO model is trained on a dataset of stacklights and persons to learn to identify them accurately. This step produces the `best.pt` weights file.
3.  **Real-time Inference (`inference.py`)**: The trained model is used to run inference on test images. This script detects objects, applies the custom alert logic, displays the results, and logs the output.
4.  **Logic Modeling (`decision_tree.py`)**: The `if-else` rules from the inference script are formally modeled using a `scikit-learn` decision tree to validate and visualize the logic.

-----

## 🚦 What are Industrial Stacklights?

An **industrial stacklight** (or signal tower light) is a common fixture on manufacturing equipment. It provides a quick visual indication of a machine's status.

  * 🟢 **Green:** Normal operation.
  * 🟡 **Amber/Orange:** Warning or fault condition.
  * 🔴 **Red:** Critical failure or emergency stop.

This project uses these color-coded signals as the primary input for determining machinery status.
<img width="1601" height="1601" alt="image" src="https://github.com/user-attachments/assets/c48f8bef-50f3-4033-b9a2-3414cf27b9af" />

-----

## 🚨 Custom Alert System

This system implements custom logic to generate real-time alerts. The `inference.py` script identifies the state and triggers alerts accordingly.

| Detected State | Machinery Status | Alert Message |
| :--- | :--- | :--- |
| 🟢 **Green Light** | Working Normally | *No alert.* |
| 🟢 **Green Light** + 👤 **Person** | Working Normally | `PERSON PRESENT ALERT!!` |
| 🟡 **Orange Light** | Faulty | `NEED A TECHNICIAN, FAULT IN MACHINERY!!` |
| 🔴 **Red Light** | Stopped | *No alert is generated.* |

-----

## 🤔 Technology: The Ultralytics YOLO Framework

This project utilizes the powerful **`ultralytics`** Python library, which is the official framework for **YOLOv11** and supports other YOLO versions.

While we might refer to our custom-trained model as 'YOLOv11' for project-specific versioning, it operates within the state-of-the-art `ultralytics` ecosystem. This approach combines the flexibility of a custom-trained model with the robust, high-performance tools provided by the `ultralytics` library for training and inference.

-----

## 📜 Key Scripts Explained

### `train.py`

This script is used to train the YOLO model on the custom dataset of stacklights and persons. After running, it saves the best performing model weights as `best.pt` inside a `runs/detect/train` directory.

### `inference.py`

This is the core execution script that simulates the monitoring system.

  * It loads the trained `best.pt` model.
  * It processes a directory of test images.
  * For each image, it performs object detection to find lights and people.
  * It implements the custom alert logic, printing alerts to the console.
  * It displays the annotated image with bounding boxes.
  * Finally, it saves all results, including timestamps and alerts, into a `stacklamp_data.csv` log file.

### `decision_tree.py`

This script serves as a formal model of our alerting logic. Instead of just implementing rules with `if-else` statements, this code uses a **Decision Tree Classifier** from `scikit-learn` to replicate the decision-making process. This helps validate and visualize the project's core logic.

-----

## ⚙️ How to Run This Project

Follow these steps to set up and run the industrial safety system on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### Step 2: Set Up the Anaconda Environment

This project uses a dedicated virtual environment to manage dependencies.

```bash
# Create a new anaconda environment
conda create --name stacklight_env python=3.9

# Activate the environment
conda activate stacklight_env
```

### Step 3: Install Dependencies

After activating your environment, install the necessary libraries. The two most important packages are **PyTorch** and **Ultralytics**.

1.  **Install PyTorch for GPU Acceleration if you are using GPU**

    PyTorch is the core deep learning framework that runs the YOLO model. This specific command installs a version compatible with NVIDIA GPUs using CUDA 11.8, which is essential for high-speed training and inference.

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

      * **Why is this needed?** This command installs PyTorch with **CUDA support**, allowing the program to use a compatible NVIDIA GPU. Running deep learning models on a GPU is significantly faster (often 10-100x) than running them on a CPU.

2.  **Install Ultralytics**

    The `ultralytics` package is the official library for the YOLO models. It provides all the tools needed to train, test, and run inference with your custom model.

    ```bash
    pip install ultralytics
    ```

      * **Why is this needed?** This library simplifies the entire computer vision pipeline. Your code uses it to load the trained model (`YOLO("best.pt")`), run predictions on images, and get bounding box information.

3.  **Install Other Libraries**

    Finally, install the remaining packages like OpenCV for image handling and Pandas for data logging. You can install them from a `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```
```

### Step 4: Run the Training (Optional)

If you have the dataset in YOLO format, you can train the model by running the training script.

```bash
python train.py
```

After training, find your weights file at `runs/detect/train/weights/best.pt`.

### Step 5: Run the Inference Script

To run the monitoring system on your test images, execute the inference script.

**Note:** Before running, open the `inference.py` script and ensure the `model_path` and `test_images_dir` variables point to the correct locations on your machine.

```bash
python inference.py
```

The script will process the images, display them one by one, print alerts in the terminal, and create a `stacklamp_data.csv` file with the results.

Of course. Here is a final section to add to the end of your `README.md` file.

***

## 📋 Summary & Training Details

This project successfully demonstrates a computer vision system for enhancing industrial workplace safety. By using a custom-trained YOLO model, the system effectively monitors machine status via stacklight colors and detects human presence, triggering context-specific alerts to prevent accidents.

The model was trained on a robust dataset to ensure accuracy in real-world scenarios.

### Training Environment & Performance
The model training was performed on a high-performance laptop with the following specifications:
* **Processor**: Intel Core i9-13950HX (13th Gen)
* **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM)
* **RAM**: 32GB

The training process for the entire dataset of **7,795 images** took approximately **7 hours** to complete on this hardware. This setup proves effective for handling the demanding task of training a deep learning model from a substantial dataset.
