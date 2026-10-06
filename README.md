# QuickBite Diner Intelligence

## Run it
1. `pip install -r requirements.txt`
2. Open `QuickBite_Analysis.ipynb` and run all cells. This rebuilds `food_models.joblib` with your own scikit-learn version.
3. In this folder: `streamlit run app.py`

## Files
- `app.py`: the interactive dashboard (seven pages, learn mode, toggles). Loads `food_models.joblib` in the format the brief asks for.
- `app_basic.py`: the original dashboard from the brief, unchanged.
- `.streamlit/config.toml`: colour theme. Keep the folder next to `app.py`.
- `diners.csv`, `orders.csv`: used by the exploration pages. Without them only the Diner predictor page runs.
