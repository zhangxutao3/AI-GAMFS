# AI-driven Global Aerosol-Meteorology Forecasting System (AI-GAMFS)


## Project Overview


AI-GAMFS is a deep learning-based model for global aerosol-meteorology coupled forecasting. It leverages PyTorch to provide high-accuracy predictions for aerosol and meteorological data, utilizing datasets such as GEOS-FP.


## Model Architecture


Below is the schematic diagram of the AI-GAMFS model architecture:


![AI-GAMFS Model Architecture](model.png)


## System Requirements


- **Operating System**: Ubuntu 22.04 (recommended for optimal compatibility and performance).
- **Hardware**:
CPU: Multi-core processor (e.g., Intel i5 or higher).
GPU: NVIDIA GPU with CUDA support (optional, for faster computation).
GPU-RAM: Minimum 24 GB.
- **Software**:
[Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Git (for cloning the repository).
**wget**: Required for downloading GEOS-FP data on Linux systems. Install it using:
```bash
sudo apt-get update
sudo apt-get install wget
```For Windows systems, wget is built into the project and does not require additional installation.

## Installation


### Step 1: Clone the Repository


To get started, clone the project to your local machine:


```bash
git clone https://github.com/zhangxutao3/AI-GAMFS.git
cd AI-GAMFS
```

### Step 2: Set Up the Python Environment


1. Create a new conda environment named gamfs:
```bash
conda create -n gamfs python=3.11
```
2. Activate the environment:
```bash
conda activate gamfs
```

### Step 3: Install Dependencies


Install all required packages (including PyTorch, xarray, etc.) using the provided requirements.txt file:


```bash
pip install -r requirements.txt
```

## Folder Structure


- **temp_asm**: Temporary storage for GEOS-FP initial field data.
- **inference**: Stores model inference results in the format 20250827_0430/AI_GAMFS.20250827_0430+20250827_0730.V01.nc, where:
20250827_0430 is the initial field timestamp.
20250827_0730 is the forecast timestamp.
V01 is the version.
AI_GAMFS is the framework name.
- **model**: Contains model files downloaded from Zenodo or Hugging Face.
- **utils**: Includes tools for model operations.

## Usage


### Step 1: Download Model Files


Download the model files from [Zenodo](https://zenodo.org/records/16810754) or [Hugging Face](https://huggingface.co/zhangxutao/AI-GAMFS) and place them in the model folder.


Available models include:


- gamfs_3h_traced.pt (4.85 GB) – For 3-hour forecasts.
- gamfs_6h_traced.pt (4.85 GB) – For 6-hour forecasts.
- gamfs_9h_traced.pt (4.85 GB) – For 9-hour forecasts.
- gamfs_12h_traced.pt (4.85 GB) – For 12-hour forecasts.

### Step 2: Prepare GEOS-FP Data


- The model uses GEOS-FP data, which supports specific time points at 3-hour intervals: 01:30, 04:30, 07:30, 10:30, 13:30, 16:30, 19:30, 22:30 (UTC).
- Ensure your input data aligns with these time points.
- **Note**: GEOS-FP data is automatically downloaded via inference.py and does not require manual download.

### Step 3: Configure Inference


Open inference.py and modify line 106 to set the desired time range:


```python
date_range = pd.date_range(
    start="2024-08-26 22:30:00",
    end="2024-08-27 22:30:00",
    freq="1D"
)
```

### Step 4: Run Inference


Execute the inference script:


```bash
python inference.py
```

### Step 5: View Results


Forecasting results are stored in the inference folder.


## Notes


- **Time Restrictions**: Always verify that time points in inference.py match GEOS-FP's 3-hourly half-hour intervals (e.g., 01:30, 04:30). Invalid time points will cause errors.
- **GPU Support**: If using a GPU, ensure the correct CUDA and cuDNN versions are installed (e.g., CUDA 11.8 for PyTorch). Check compatibility with PyTorch.
- **Data Formats**: GEOS-FP data is typically in NetCDF format. Use xarray or netCDF4 for processing.
- **Environment Management**: Modifying the base environment may affect other projects. Use the dedicated gamfs environment to avoid conflicts.

## Citation


If you use AI-GAMFS in your research, please cite the following paper:


```bibtex
@misc{gui2024advancingglobalaerosolforecasting,
      title={Advancing global aerosol forecasting with artificial intelligence},
      author={Ke Gui and Xutao Zhang and Huizheng Che and Lei Li and Yu Zheng and Linchang An and Yucong Miao and Hujia Zhao and Oleg Dubovik and Brent Holben and Jun Wang and Pawan Gupta and Elena S. Lind and Carlos Toledano and Hong Wang and Zhili Wang and Yaqiang Wang and Xiaomeng Huang and Kan Dai and Xiangao Xia and Xiaofeng Xu and Xiaoye Zhang},
      year={2024},
      eprint={2412.02498},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph},
      url={https://arxiv.org/abs/2412.02498},
}
```
