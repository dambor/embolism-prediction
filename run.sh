#!/bin/bash
# Run script for the embolism prediction model

# Create virtual environment if it doesn't exist
if [ ! -d "embolism_env" ]; then
    echo "Creating virtual environment..."
    python -m venv embolism_env
fi

# Activate virtual environment
source embolism_env/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Ensure all necessary files are in their proper locations
echo "Checking file structure..."

# Create directories
mkdir -p data results

# Check if simplified_embolism_model.py exists in the root directory
if [ ! -f "simplified_embolism_model.py" ]; then
    # If it exists in models directory, copy it to root
    if [ -f "models/simplified_embolism_model.py" ]; then
        echo "Moving simplified_embolism_model.py to root directory..."
        cp models/simplified_embolism_model.py .
    fi
fi

# Check if download_demo_data.py exists in the root directory
if [ ! -f "download_demo_data.py" ]; then
    # If it exists in data directory, copy it to root
    if [ -f "data/download_demo_data.py" ]; then
        echo "Moving download_demo_data.py to root directory..."
        cp data/download_demo_data.py .
    fi
fi

# Run the model
echo "Running embolism prediction model..."
python main.py --data_dir ./data --output_dir ./results --max_patients 2000000 --epochs 15000 --batch_size 16 --patience 3

echo "Done! Results saved to ./results"