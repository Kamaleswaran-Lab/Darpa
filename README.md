# Bootstrap Ensemble Submission - OCA-SENTINEL

Optimized submission package with only files required to generate age group predictions.

## Files Included

### Core Scripts
- `generate_submission.py` - Main prediction generation script

### Models
- `checkpoints/` - 10 pre-trained bootstrap models (206 MB total)
  - `best_model_bootstrap_1.pth` through `best_model_bootstrap_10.pth`

### Test Data
- `aortaP_test_data.csv` - Aorta pressure waveform features (875 samples × 336 features)
- `brachP_test_data.csv` - Brachial pressure waveform features (875 samples × 336 features)

### Model Code & Config
- `src/` - Model architecture and data utilities
- `config/` - Training configuration files
- `requirements.txt` - Python dependencies

### Output
- `OCA-SENTINEL_output.json` - Generated predictions (875 samples)

## Usage

Generate predictions:
```bash
python generate_submission.py
```

Custom parameters:
```bash
python generate_submission.py \
  --checkpoint_dir checkpoints \
  --aorta_file aortaP_test_data.csv \
  --brach_file brachP_test_data.csv \
  --output OCA-SENTINEL_output.json \
  --method averaging \
  --device cpu \
  --batch_size 8
```

## Output Format

JSON file with 875 predictions mapping sample indices to age group classes:
```json
{
  "0": 0,
  "1": 0,
  "2": 0,
  ...
  "874": 0
}
```

Classes: 0=20s, 1=30s, 2=40s, 3=50s, 4=60s, 5=70s

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```


