# Heart Disease Prediction Project
This project predicts heart disease using machine learning algorithms.
Uses the South african heart disease dataset from openml.

## Dependencies

### System Dependencies
- python 3.7 or above

### Python Dependencies
- scikit-learn
- pandas
- tensorflow
- requests
- matplotlib

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Usage
```python
# To prepare the data
python3 prepare_data.py

# To train and compile the model
python3 train_and_compile.py

# To predict using the model
python3 predict.py
```
