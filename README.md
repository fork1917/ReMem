# ReMem: A Dynamic Memory Evolution Detector for Zero-Shot Anomaly Detection



This is the official implementation of the paper **"ReMem: A Dynamic Memory Evolution Detector for Zero-Shot Anomaly Detection"**.



## 🛠️ Installation



To reproduce the experiments, please create a virtual environment and install the required dependencies.

1. **Create and activate the environment**

   Bash

   ```bash
   conda create -n remem python=3.10
   conda activate remem
   ```

2. **Install dependencies**

   Bash

   ```bash
   pip install -r requirements.txt
   ```



## 📂 Dataset Preparation



Download the **MVTec-AD** and **VisA** datasets from their official sources.

- **MVTec-AD:** [Download Link](https://www.mvtec.com/company/research/datasets/mvtec-ad) (Default path: `data/mvtec_anomaly_detection`)
- **VisA:** [Download Link](https://github.com/amazon-science/spot-diff) (Default path: `data/VisA_pytorch/1cls/`)



### Directory Structure



Ensure your data is organized as follows (or update the paths in `Config.py`):

Plaintext

```bash
your_data_root
├── object1
│   ├── ground_truth        # Anomaly annotations (masks)
│   │   ├── anomaly_type1
│   │   └── ...
│   ├── test                # Test images (anomalies & good)
│   │   ├── anomaly_type1
│   │   ├── ...
│   │   └── good
│   └── train               # Train/Reference images (nominal only)
│       └── good
├── object2
│   ├── ...
```



### VisA Dataset Preprocessing



For the VisA dataset, please follow the official repository instructions to organize the data into the 1-class split. Once downloaded, run the provided transformation script to adapt the format for ReMem:

Bash

```bash
# Ensure you change the path inside visa_transform.py before running
python visa_transform.py
```



## 🚀 Usage





### 1. Configuration



Modify **`Config.py`** to set your specific dataset paths and parameters.

> **Note on Pre-computed Results:** The `.json` files included in this project store the cached results/initializations used by the ReMem launch method. If you wish to replace these with your own results, please ensure your new files strictly follow the same JSON format.



### 2. Running the Model



Once the environment is set and data is prepared, run the main execution script:

Bash

```
python ReRem_run.py
```