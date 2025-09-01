# FCS-Brownian-Simulator
Interactive Fluorescence Correlation Spectroscopy (FCS) simulation tool visualizing Brownian motion of particles through a confocal volume with real-time analysis


## Overview
This is a Fluorescence Correlation Spectroscopy (FCS) simulation tool that visualizes particle diffusion through a confocal volume in real-time using PyQt and PyQtGraph with OpenGL acceleration.

## Prerequisites
- Python 3.7 or higher (3.8+ recommended)
- Windows, macOS, or Linux operating system

## Installation

### Step 1: Install Python
If you don't have Python installed:
1. Download Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Verify installation: `python --version`

### Step 2: Install Required Packages

Open a terminal/command prompt and run:

```bash
pip install numpy pyqtgraph PyQt6 PyQt6-Qt6 opencv-python qdarkstyle
```

## Running the Script

### Method 1: Command Line
Navigate to the script directory and run:
```bash
python fcs_pyqtgraph.py
```


## Features and Controls

### Main Controls Tab
- **Concentration (nM)**: Adjust particle concentration (0.1-100 nM)
- **Box Size Factor**: Control simulation volume size (2-10x)
- **Radius (nm)**: Set particle hydrodynamic radius (1-20 nm)
- **Time Step (μs)**: Simulation time resolution (1-500 μs)
- **Pause/Resume**: Pause the simulation
- **Reset**: Reset simulation to initial state
- **2D Mode**: Switch between 2D (particles on plane) and 3D simulation
- **Show Trails**: Toggle particle trajectory visualization

### Performance Settings
- **Performance Dropdown**: 
  - Balanced: Default settings
  - High Quality: Better visuals, lower FPS
  - Max Speed: Maximum performance
- **Max Trails**: Limit number of particle trails (1-100)
- **FPS Display**: Shows current frame rate

### Two-Particle Mode Tab
- Enable simulation of two different particle types
- Configure separate concentrations, sizes, and brightness
- View separate intensity traces for each type

### Video Recording Tab
- **Start/Stop Recording**: Capture simulation as video
- **Quality**: Low/Medium/High bitrate options
- **FPS**: Video frame rate (15-60 fps)
- Videos saved to `fcs_videos/` folder

## Understanding the Display

### 3D Visualization (Left Panel)
- **Green/Blue particles**: Inside confocal volume
- **Gray particles**: Outside detection region
- **Colored mesh**: Confocal volume intensity profile
- **Grid lines**: Reference planes
- **Box boundaries**: Simulation volume limits

### Plots (Right Panel)
1. **Fluorescence Intensity**: Real-time intensity fluctuations
2. **Autocorrelation G(τ)**: Correlation analysis showing diffusion timescale

### Status Bar
Shows particle count, diffusion coefficient, characteristic time, and simulation time

## Troubleshooting

### Import Errors
If you get module import errors:
```bash
pip install --upgrade pip
pip install --force-reinstall numpy pyqtgraph PyQt6 PyQt6-Qt6 opencv-python
```

Note: The script uses `pyqtgraph.Qt` which automatically detects and uses the available Qt binding (PyQt6, PyQt5, or PySide). PyQt6 is recommended for better performance and modern features.

### OpenGL Issues
If you encounter OpenGL errors:
1. Update graphics drivers
2. Try software rendering:
```bash
set QT_OPENGL=software  # Windows
export QT_OPENGL=software  # Linux/Mac
python fcs_pyqtgraph.py
```

### Performance Issues
- Use "Max Speed" performance mode
- Reduce "Max Trails" to 10-20
- Lower particle concentration
- Disable trails completely

### Video Recording Issues
If video recording fails:
- Ensure `fcs_videos/` folder has write permissions
- Try different quality settings
- Check available disk space

## Scientific Background
This simulation models:
- Brownian motion of fluorescent particles
- 3D Gaussian confocal detection volume
- Fluorescence intensity fluctuations
- Autocorrelation analysis for diffusion measurements

The simulation uses the Stokes-Einstein equation for diffusion and proper periodic boundary conditions.

