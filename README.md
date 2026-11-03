## **The official implementation of MLSTTrack: Modality-adaptive Language-guided Sparse Temporal learning framework for RGB-T Tracking**

!["Framework"](/fig/模型结构图1.png)

## ⚙️ Installation

**1. Clone the repository and create the conda environment:**

```bash
git clone https://github.com/IdolLab/MLSTTrack.git
cd MLSTTrack
conda create -n MLSTTrack python=3.10
conda activate MLSTTrack
```
**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 📁 Data Preparation

Download the following datasets and place them under `./data/`:

- **RGB-T images**: [GTOT, RGBT210, RGBT234, LasHeR](https://chenglongli.cn/Datasets-and-benchmark-code/)
- **Textual annotations**: [Here](https://pan.baidu.com/s/1H2ZSTpCElm9B0WswTDkzRA?pwd=61h4)  (pwd: `61h4`)  Thanks for [RAGTrack](https://github.com/IdolLab/RAGTrack).
- **The experimental results of the four datasets were calculated using the [Matlab tool](https://chenglongli.cn/Datasets-and-benchmark-code/) and are presented in the "[experiment_result](https://github.com/yangkang121/MLSTTrack/tree/main/experiment_result)" file.**

The expected directory structure is as follows:

```text
MLSTTrack/
└── data/
    ├── GTOT/
    │   ├── BlackCar/
    │   │   ├── i/
    │   │   ├── v/
    │   │   ├── groundTruth_i.txt
    │   │   ├── groundTruth_v.txt
    │   │   ├── visible_description.txt
    │   │   └── class.txt
    │   └── ...
    ├── RGBT210/
    │   ├── afterrain/
    │   │   ├── infrared/
    │   │   ├── visible/
    │   │   ├── init.txt
    │   │   ├── visible_description.txt
    │   │   └── class.txt
    │   └── ...
    ├── RGBT234/
    │   └── ...  (same structure as RGBT210)
    └── LasHeR/
        ├── train/
        │   ├── 1boygo/
        │   │   ├── infrared/
        │   │   ├── visible/
        │   │   ├── init.txt
        │   │   └── visible_description.txt
        │   └── ...
        └── test/
            ├── 1blackteacher/
            │   ├── infrared/
            │   ├── visible/
            │   ├── init.txt
            │   ├── visible_description.txt
            │   └── class.txt
            └── ...
```

---

## 🔧 Setup & Configuration

Run the following command to initialize local paths:

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

Alternatively, manually edit the path configs:

```text
./lib/train/admin/local.py      # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

---

This repo is based on [DUTrack](https://github.com/GXNU-ZhongLab/DUTrack) and [RAGTrack](https://github.com/IdolLab/RAGTrack). We sincerely thank the authors for their excellent works.

If you have any questions, please feel free to contact me at any time: yangk19990915@163.com

---

## 

```bibtex


```

