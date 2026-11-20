# CSE 242 Assignment 4

Extract the entire archive into one directory. From that directory, run:

```sh
python -m pip install numpy scikit-learn matplotlib pillow
python experiment.py
```

Running the script fits the models and saves four RGB figures beside the source: `choosing_k.png`, `cluster_profiles.png`, `repair_by_seed.png`, and `repair_outcomes.png`. The experiment uses fixed seeds for the splits, restarts, and corruption draws.

## Data

Both experiments use the 1,797-image digits dataset bundled with scikit-learn. No data download or API key is needed. This is the test partition of the original UCI dataset. Part 3 makes its own stratified training, validation and test split within those 1,797 images.

- E. Alpaydin and C. Kaynak (1998), [Optical Recognition of Handwritten Digits, UCI Machine Learning Repository](https://doi.org/10.24432/C50P49).
- [scikit-learn digits dataset and loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

## Acknowledgments

NumPy supplies array operations and random sampling. Scikit-learn supplies the data, scaling, evaluation metrics, splitting, nearest-neighbor search and SVM. The K-means implementation is in `experiment.py`. Matplotlib renders the figures. Pillow writes the RGB PNGs.
