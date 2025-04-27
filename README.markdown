# Gait Analysis for FOG Detection

This project analyzes gait data to identify key biomarkers (ankle, thigh, trunk) influencing Freezing of Gait (FOG) using the Daphnet FOG dataset and a normal dataset. It trains Random Forest, XGBoost, Logistic Regression, and LSTM models, evaluates them, and visualizes feature importance.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gait-analysis.git
   cd gait-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place Daphnet FOG dataset files in `data/daphnet/` and normal dataset files in `data/normal/`.
4. Run the main script:
   ```bash
   python main.py
   ```

## Project Structure
- `preprocess.py`: Loads and preprocesses Daphnet FOG and normal datasets.
- `models.py`: Trains and evaluates models, analyzes feature importance.
- `main.py`: Executes the full pipeline.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Notes
- Update file paths in `main.py` for normal dataset files.
- Outputs include classification reports, feature importance plots, and SHAP visualizations.
- Daphnet dataset is publicly available; normal dataset must be provided by the user.

## License
MIT License