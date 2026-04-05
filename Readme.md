# ITM Signal Estimation

This repository encapsulates the Longley-Rice (ITM) model terrain and signal estimation workflow. It automates the following tasks with a given target area latitude and longitude:
1. **DEM Downloading**
2. **Grid-based Path Loss Generation**
3. **GIS Overlay Mapping**

## Setup Instructions

This project relies on strict system-level geospatial C++ dependencies. **Do not use `pip install` directly for the data science packages**. Please use Anaconda/Miniconda to guarantee OS compatibility across Mac and Windows.

### 1. **Clone the repository and its submodule:**
   ```bash
   git clone <YOUR_REPO_URL>
   cd itm_signal_estimation
   ```

### 2. Install `itmlogic`
We use [itmlogic](https://github.com/edwardoughton/itmlogic) to compute Longley-Rice (ITM) path loss. Clone and install from source:

```bash
git clone https://github.com/edwardoughton/itmlogic.git
```

### 3. Create Python Environment with Conda

```bash
conda env create -f environment-itm.yml
```



## Usage

Run `main.py` to compute the estimated RSSI for the test region:
```bash
python main.py
```

The result should be like:

![Signal Estimation Test Result](test_result.png)


