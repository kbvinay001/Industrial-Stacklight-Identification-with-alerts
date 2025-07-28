# Industrial Safety System with Stacklight & Person Detection 🚦👷

This project uses computer vision to enhance industrial safety by monitoring machinery status through stacklight colors and detecting human presence in proximity to active equipment. It leverages a custom-trained **YOLO** model within the `ultralytics` framework to identify stacklight colors (simulated with traffic lights) and persons, triggering specific alerts and logging the data.

-----

## 🚀 Project Workflow

This project follows a clear, sequential workflow from setup to execution:

1.  **Environment Setup**: A dedicated virtual environment is created using **Anaconda** to manage all project dependencies.
2.  **Dependency Installation**: Key libraries like PyTorch (with GPU support) and Ultralytics are installed.
3.  **Model Training (`train.py`)**: A custom YOLO model is trained on the dataset, producing the `best.pt` weights file.
4.  **Real-time Inference (`inference.py`)**: The trained model is used to run inference on test images to apply the custom alert logic.
5.  **Logic Modeling (`decision_tree.py`)**: The alert logic is formally modeled and run using a `scikit-learn` decision tree to validate and visualize the rules.

-----

## 🚦 What are Industrial Stacklights?

An **industrial stacklight** (or signal tower light) is a common fixture on manufacturing equipment. It provides a quick visual indication of a machine's status.

  * 🟢 **Green:** Normal operation.
  * 🟡 **Amber/Orange:** Warning or fault condition.
  * 🔴 **Red:** Critical failure or emergency stop.

This project uses these color-coded signals as the primary input for determining machinery status.
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/cdec357a-6abf-4359-8514-52c998cf86db" />


-----

## 🚨 Custom Alert System

This system implements custom logic to generate real-time alerts. The `inference.py` script identifies the state and triggers alerts accordingly.

| Detected State | Machinery Status | Alert Message |
| :--- | :--- | :--- |
| 🟢 **Green Light** | Working Normally | *No alert.* |
| 🟢 **Green Light** + 👤 **Person** | Working Normally | `PERSON IS PRESENT NEAR THE MACHINERY, ALERT!!` |
| 🟡 **Orange Light** | Faulty | `NEED A TECHNICIAN, FAULT IN MACHINERY!!` |
| 🔴 **Red Light** | Stopped | *No alert is generated.* |

-----

## 🤔 Technology: The Ultralytics YOLO Framework

This project utilizes the powerful **`ultralytics`** Python library, which is the official framework for **YOLOv8** and supports other YOLO versions. While we might refer to our custom-trained model as 'YOLOv11' for project-specific versioning, it operates within the state-of-the-art `ultralytics` ecosystem. This approach combines the flexibility of a custom-trained model with the robust, high-performance tools provided by the `ultralytics` library for training and inference.

-----

## 📜 Key Scripts Explained

### `train.py`

This script is used to train the YOLO model on the custom dataset of stacklights and persons. After running, it saves the best performing model weights as `best.pt` inside a `runs/detect/train` directory.

### `inference.py`

This is the core execution script that simulates the monitoring system. It loads the `best.pt` model, processes a directory of test images, applies the custom alert logic, displays results, and saves a log to `stacklamp_data.csv`.

### `decision_tree.py`

This script serves as a formal model of our alerting logic. It uses a **Decision Tree Classifier** from `scikit-learn` to replicate the decision-making process, which helps validate and visualize the project's core logic.

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

After activating your environment, install the necessary libraries.

1.  **Install PyTorch for GPU Acceleration**
    PyTorch is the core deep learning framework that runs the YOLO model. This command installs a version compatible with NVIDIA GPUs (CUDA 11.8), which is essential for high-speed performance.

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

2.  **Install Ultralytics**
    The `ultralytics` package is the official library for YOLO models, providing all tools needed to train, test, and run inference.

    ```bash
    pip install ultralytics
    ```

3.  **Install Other Libraries**
    Finally, install the remaining packages like OpenCV and Pandas from a `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### Step 4: Train the Model (Optional)

If you have the dataset in YOLO format, you can train the model by running the training script. After training, find your weights file at `runs/detect/train/weights/best.pt`.

```bash
python train.py
```

### Step 5: Run the Inference Script

To run the monitoring system on your test images, execute the inference script.
**Note:** Before running, open `inference.py` and ensure the `model_path` and `test_images_dir` variables point to the correct locations.

```bash
python inference.py
```

The script will process the images, display them, print alerts, and create a `stacklamp_data.csv` file.

### Step 6: Run the Decision Tree

To run the script that models the alert logic using a decision tree, execute the following command. This script typically reads the `stacklamp_data.csv` generated by the inference script.

```bash
python decision_tree.py
```

-----

## 📋 Summary & Training Details

This project successfully demonstrates a computer vision system for enhancing industrial workplace safety. By using a custom-trained YOLO model, the system effectively monitors machine status via stacklight colors and detects human presence, triggering context-specific alerts.

### Training Environment & Performance

The model training was performed on a high-performance laptop with the following specifications:

  * **Processor**: Intel Core i9-13950HX (13th Gen)
  * **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM)
  * **RAM**: 32GB

The training process for the entire dataset of **7,795 images** took approximately **7 hours** to complete on this hardware. This setup proves effective for handling the demanding task of training a deep learning model from a substantial dataset.
