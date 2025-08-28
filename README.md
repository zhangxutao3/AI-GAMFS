# AI-driven Global Aerosol-Meteorology Forecasting System (AI-GAMFS)

## Project Overview
AI-GAMFS is a deep learning-based model for global aerosol-meteorology coupled forecasting. It leverages PyTorch to provide high-accuracy predictions for aerosol and meteorological data, utilizing datasets such as GEOS-FP.

## Model Architecture
Below is the schematic diagram of the AI-GAMFS model architecture:

![AI-GAMFS Model Architecture](model.png)

## Getting Started
To begin, clone this project to your local machine:
```bash
git clone https://github.com/zhangxutao3/AI-GAMFS.git
cd AI-GAMFS
```

## Environment Setup
To set up the project environment, ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

1. **Configure the `base` environment**:
   The project uses the `base` Anaconda environment with Python 3.11.4. Update it with the provided `environment.yml`:
   ```bash
   conda env update -n base -f environment.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate base
   ```

## Dependencies
The project requires the following dependencies (included in `environment.yml`):
- Python 3.11.4
- `pandas` (for data manipulation, e.g., time range generation)
- `numpy` (for numerical computations)
- `pytorch` (deep learning framework)
- `xarray` (for handling NetCDF files, common in meteorological data)
- `netCDF4` (for GEOS-FP data processing)
- `matplotlib` and `seaborn` (for visualization)
- `h5py` (optional, for HDF5 file handling if needed)

To manually install dependencies in the `base` environment:
```bash
conda install pandas numpy xarray netCDF4 matplotlib seaborn h5py
# Install PyTorch (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# Or PyTorch (GPU, if supported)
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
```

## Usage
1. **Download Model Files**:
   - Download the model files from [Zenodo](https://zenodo.org/records/16810754) and place them in the `model` folder.

2. **Prepare GEOS-FP Data**:
   - The model uses GEOS-FP data, which only supports specific time points at 3-hour intervals: 01:30, 04:30, 07:30, 10:30, 13:30, 16:30, 19:30, 22:30 (UTC).
   - Ensure your input data aligns with these time points.
   - ***GEOS-FP data is automatically downloaded through inference.py and does not require manual download.***

3. **Configure Inference**:
   - Open `inference.py` and modify line 106 to set the desired time range:
     ```python
     date_range = pd.date_range(
         start="2024-08-26 22:30:00",
         end="2024-08-27 22:30:00",
         freq="1D"
     )
     ```

4. **Run Inference**:
   ```bash
   python inference.py
   ```

5. **Check Results**:
   - Forecasting results are stored in the `inference` folder.

## Notes
- **Time Restrictions**: Always verify that time points in `inference.py` match GEOS-FP's 3-hourly half-hour intervals (e.g., 01:30, 04:30). Invalid time points will cause errors.
- **GPU Support**: If using a GPU, ensure the correct CUDA and cuDNN versions are installed (e.g., CUDA 11.8 for PyTorch). Check compatibility with PyTorch.
- **Data Formats**: GEOS-FP data is typically in NetCDF format. Use `xarray` or `netCDF4` for processing.
- **Base Environment**: Modifying the `base` environment may affect other projects. Consider creating a dedicated environment (e.g., `conda create -n ai-gamfs`) to avoid conflicts.

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
```