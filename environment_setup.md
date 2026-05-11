## Environment Setup

The environment used for developing, training, and testing the crowd behaviour analysis system is as follows:

###  Hardware Configuration:
- **Processor**: AMD Ryzen 7 5800HS
- **RAM**: 16 GB
- **Storage**: SSD - 512 GB
- **Graphics**: NVIDIA RTX 3050 Laptop GPU

### Software & Tools:
- **Operating System**: Windows 11
- **Programming Language**: Python 3.12.7
- **IDE/Editor**: PyCharm
- **Libraries and Frameworks**:
  - `numpy`, `pandas` – data processing
  - `opencv-python` – video and frame handling
  - `ultralytics` - YOLO detection model
  - `torch+cuda` - to run YOLO model using GPU
  - `scikit-learn` – machine learning models (Random Forest)
  - `matplotlib`, `seaborn` – data visualization
  - `joblib` – model serialization
- **Other Packages**:
  - `json`, `os` – file and label handling
  - `sklearn.metrics` – for accuracy, precision, recall, F1-score

### Execution Environment:
- The project was executed on the above configuration using a virtual environment.
- To set up the environment:
  ```
  python -m venv crowd-env
  crowd-env\Scripts\activate
  pip install -r requirements.txt