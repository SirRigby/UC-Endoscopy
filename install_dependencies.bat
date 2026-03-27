@echo off
echo Installing required Python packages...


python -m pip install --upgrade pip


pip install torch torchvision torchaudio


pip install opencv-python pillow


pip install tqdm


pip install numpy

echo.
echo All dependencies installed successfully!
pause