cd Selection
echo "Running Select k Best"
python SelectKBest.py
echo "Running Tree-based Selection"
python TreeBasedSelection.py
echo "Running Variance Threshold"
python VarianceThreshold.py
