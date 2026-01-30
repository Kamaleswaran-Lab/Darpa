# DARPA Age Group Prediction - Submission Package

## Contents

This submission package contains all necessary files to generate predictions on the test dataset.

### Files Structure
```
submission/
├── generate_submission.py     # Main prediction script
├── best_model.pth            # Trained model checkpoint
├── aortaP_test_data.csv      # Test data (AortaP signals)
├── brachP_test_data.csv      # Test data (BrachP signals)
├── requirements.txt          # Python dependencies
├── config/
│   └── config.yaml          # Model configuration
└── src/
    ├── data/
    │   ├── dataset.py       # Dataset preprocessing
    │   └── preprocessing.py # Signal filtering utilities
    ├── models/
    │   └── age_group_predictor.py  # Model architecture
    └── utils/
        └── config.py        # Configuration utilities
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- PyYAML

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Predictions

```bash
python generate_submission.py \
    --model best_model.pth \
    --aorta-data aortaP_test_data.csv \
    --brach-data brachP_test_data.csv \
    --output OCA-SENTINEL_output.json \
    --device auto
```

### Arguments

- `--model`: Path to trained model checkpoint (default: `best_model.pth`)
- `--aorta-data`: Path to AortaP test data CSV
- `--brach-data`: Path to BrachP test data CSV
- `--output`: Output JSON file path
- `--device`: Device for inference (`auto`, `cuda`, or `cpu`)
- `--config`: Path to config file (default: `config/config.yaml`)
- `--team-name`: Team name for filename validation (`OCA-SENTINEL`)

### Device Options

- `auto` (default): Automatically detects GPU availability
- `cuda`: Force GPU usage (falls back to CPU if unavailable)
- `cpu`: Force CPU usage

### Output Format

The script generates a JSON file with predictions in the format:
```json
{
  "0": 2,
  "1": 4,
  "2": 1,
  ...
}
```

Where keys are sample indices (0-874) and values are predicted age group classes (0-5).

## Model Architecture

- **Encoder**: Transformer-based encoders for AortaP and BrachP signals
- **Features**: Handles missing data with attention masking
- **Preprocessing**: Butterworth low-pass filtering + normalization
- **Classes**: 6 age groups (0-5)

## Notes

- The script automatically preprocesses signals (filtering + normalization)
- Predictions are validated before saving
- Progress updates are shown during inference
- Class distribution statistics are displayed after generation
