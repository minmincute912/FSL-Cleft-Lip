# FSL_Cleft_Lip
Training and Inference for Recognizing the Cleft Lip in the Fetal 3D Ultrasound Image by Using Few Shot Learning
=======

This repository implements few-shot learning techniques, such as **Prototypical Networks** and **Matching Networks**, to classify fetal 3D ultrasound images into two categories: `cleft` and `non-cleft`. It leverages PyTorch and includes complete code for training, evaluating, and inferring new query data.

---

## Features

- Implements **Prototypical Networks** and **Matching Networks** for few-shot learning.
- Includes support for **custom datasets** like cleft lip classification.
- Modular and extensible design for datasets, models, and training pipelines.
- GPU-accelerated with PyTorch.
- Code for **training**, **evaluation**, and **inference** included.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Usage](#usage)
   - [Training](#training)
   - [Inference](#inference)
4. [Configuration](#configuration)
5. [Contributing](#contributing)
6. [License](#license)

---

## Installation

### **Requirements**

- Python 3.8+
- PyTorch 1.11+
- torchvision
- scikit-learn
- tqdm
- PIL (Pillow)

### **Setup Environment**

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/few-shot-cleft-lip.git
    cd FSL_Cleft_Lip
2. Create a virtual environment:
    ```bash
    python3 -m venv cleftlip
    source cleftlip/bin.activate
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt

## Dataset Preparation

### Dataset Structure

Ensure your dataset follows this structure:
```bash
fetal_3dultrasound/
├── cleftlip_data/
│   ├── images_background/
│   │   ├── cleft_group/
│   │   │   ├── img1.png
│   │   │   ├── img2.png
│   │   ├── non_cleft_group/
│   │       ├── img3.png
│   │       ├── img4.png
│   ├── images_evaluation/
│       ├── cleft_group/
│       ├── non_cleft_group/
├── test/
    ├── query1.png
    ├── query2.png
```
## Usage

### Training

Train the Prototypical Network
```bash
   python experiments/proto_nets.py \
   --dataset cleft_lip \
    --n-train 1 --k-train 2 --q-train 15 \
    --n-test 1 --k-test 2 --q-test 1 \
    --distance l2
```
### Inference

Perform inference using a trained model
```bash
    python inference.py \
    --support_dir /path/to/images_background \
    --query_dir /path/to/test \
    --checkpoint /path/to/checkpoint.pth \
    --distance l2
```

## Configuration

Update the 'config.py' file for paths and global settings:


```python
# Update the dataset and model paths
DATA_PATH = "/path/to/fetal_3dultrasound"
PATH = "/path/to/save/models"
```
## Contributing 

Contributions are welcome! To contribute:
1. Fork this repository
2. Create a feature branch:
```bash
git checkout -b feature-name
```
3. Commit changes:
```bash
git commit -m "Add feature"
```
4. Push changes and create a pull request.

## License

This project is licensed under the MIT License. See the 'LICENSE' for details.
Feel free to contact [dominh9122002@gmail.com] for questions of further clarifications!
