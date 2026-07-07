# EDA Notebook

Open Jupyter and create `notebooks/eda.ipynb`.

Use helpers from `src/preprocess.py`:
- `load_data(path)` — load & clean the CSV
- `split_features_target(df)` — get X, y
- `build_preprocessor()` — sklearn ColumnTransformer

## Suggested cells
1. Load data, display `.head()` and `.info()`
2. Missing value heatmap
3. Histograms for numeric features
4. Class distribution bar chart
5. Correlation heatmap
6. Pairplot by target
