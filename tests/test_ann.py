import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add the parent directory to sys.path so we can import  our project modules without needing a package install.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing import load_data, check_missing_values, select_features, encode_labels,normalize_features, split_data,ADMIN_COLS,TARGET_COL, CLASS_ORDER

from model import build_model, NUM_CLASSES
from evaluate import evaluate_model, get_predictions, analyze_by_risk_level



DATASET_PATH = os.path.join(ROOT, 'data', 'dataset.csv')


@pytest.fixture(scope='session')
def raw_df():
    return load_data(DATASET_PATH)  #Load the raw dataset once for the entire test session.



@pytest.fixture(scope='session')
def features_and_labels(raw_df):
    X, y = select_features(raw_df) #Extract the feature matrix X and raw label series y.
    return X, y


@pytest.fixture(scope='session')
def encoded_labels(features_and_labels):
    """One-hot encode the label series and return encoder too."""
    X, y = features_and_labels
    y_encoded, le = encode_labels(y)
    return X, y_encoded, le


@pytest.fixture(scope='session')
def split_arrays(encoded_labels):
    """Split the encoded data into train / val / test."""
    X, y_encoded, le = encoded_labels
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)
    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture(scope='session')
def scaled_arrays(split_arrays, tmp_path_factory):
    X_train, X_val, X_test, y_train, y_val, y_test = split_arrays
    scaler_path = str(tmp_path_factory.mktemp('artefacts') / 'test_scaler.pkl') #Scale the split data and save the scaler to a temp directory.
    X_train_s, X_val_s, X_test_s, scaler = normalize_features(
        X_train.values, X_val.values, X_test.values,
        scaler_save_path=scaler_path
    )
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler


@pytest.fixture(scope='session')
def small_trained_model(scaled_arrays):
    """
    Build and train a model for just 5 epochs.
    This fixture exists purely to test evaluation functions. It is not meant to produce a clinically useful model.
    Training for 5 epochs is enough to confirm the model compiles, trains, and produces valid
    probability outputs without making the test suite slow.
    """
    X_train_s, X_val_s, _, y_train, y_val, _, _ = scaled_arrays
    model = build_model(input_dim=X_train_s.shape[1])
    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=5,
        batch_size=32,
        verbose=0   # Suppress output during tests
    )
    return model



class TestDataLoading:

    def test_dataset_file_exists(self):
        """Dataset CSV must exist at the expected path."""
        assert os.path.exists(DATASET_PATH), (
            f"Dataset not found at {DATASET_PATH}. "
            f"Place the CSV in the 'data/' folder."
        )

    def test_loads_as_dataframe(self, raw_df):
        """load_data() should return a non-empty pandas DataFrame."""
        assert isinstance(raw_df, pd.DataFrame)
        assert raw_df.shape[0] > 0, "Loaded DataFrame is empty."

    def test_expected_row_count(self, raw_df):
        """Dataset should contain exactly 1,000 patient records."""
        assert raw_df.shape[0] == 1000, (
            f"Expected 1000 rows, got {raw_df.shape[0]}. "
            f"The source CSV may have been modified."
        )

    def test_target_column_present(self, raw_df):
        """The 'Level' column must be present in the raw dataset."""
        assert TARGET_COL in raw_df.columns, (
            f"Target column '{TARGET_COL}' not found. "
            f"Columns present: {raw_df.columns.tolist()}"
        )

    def test_all_three_target_classes_present(self, raw_df):
        """Raw dataset must contain Low, Medium, and High samples."""
        classes_found = set(raw_df[TARGET_COL].unique())
        expected      = set(CLASS_ORDER)
        assert expected.issubset(classes_found), (
            f"Missing classes: {expected - classes_found}"
        )



class TestMissingValues:

    def test_no_missing_in_features(self, features_and_labels):
        """Feature matrix should have zero missing values after selection."""
        X, _ = features_and_labels
        total_missing = X.isnull().sum().sum()
        assert total_missing == 0, (
            f"Found {total_missing} missing values in features. "
            f"Preprocessing must handle these before training."
        )

    def test_no_missing_in_labels(self, features_and_labels):
        """Label series should have zero missing values."""
        _, y = features_and_labels
        assert y.isnull().sum() == 0, "Found missing values in labels."

    def test_missing_value_audit_returns_series(self, raw_df):
        """check_missing_values() must return a pandas Series."""
        result = check_missing_values(raw_df)
        assert isinstance(result, pd.Series)



class TestFeatureSelection:

    def test_correct_feature_count(self, features_and_labels):
        """
        After dropping 'index', 'Patient Id', and 'Level',
        exactly 23 feature columns should remain.
        """
        X, _ = features_and_labels
        assert X.shape[1] == 23, (
            f"Expected 23 features, got {X.shape[1]}. "
            f"Check ADMIN_COLS and TARGET_COL in preprocessing.py."
        )

    def test_admin_columns_removed(self, features_and_labels):
        """
        Administrative columns ('index', 'Patient Id') carry no
        clinical signal and must not appear in the feature matrix.
        """
        X, _ = features_and_labels
        for col in ADMIN_COLS:
            assert col not in X.columns, (
                f"Admin column '{col}' found in feature matrix. "
                f"It should have been dropped in select_features()."
            )

    def test_target_column_removed_from_features(self, features_and_labels):
        """'Level' must not appear as a feature as it is the target."""
        X, _ = features_and_labels
        assert TARGET_COL not in X.columns

    def test_all_features_are_numeric(self, features_and_labels):
        """
        All feature columns must be numeric (int or float).
        Non-numeric columns would break normalization silently.
        """
        X, _ = features_and_labels
        non_numeric = [c for c in X.columns
                       if not pd.api.types.is_numeric_dtype(X[c])]
        assert len(non_numeric) == 0, (
            f"Non-numeric columns found: {non_numeric}"
        )

    def test_label_series_is_string_type(self, features_and_labels):
        """Labels should be string ('Low'/'Medium'/'High') before encoding."""
        _, y = features_and_labels
        assert y.dtype == object or pd.api.types.is_string_dtype(y)


class TestLabelEncoding:

    def test_one_hot_has_three_columns(self, encoded_labels):
        """One-hot encoding must produce 3 columns (one per class)."""
        _, y_encoded, _ = encoded_labels
        assert y_encoded.shape[1] == NUM_CLASSES, (
            f"Expected {NUM_CLASSES} columns in one-hot array, "
            f"got {y_encoded.shape[1]}."
        )

    def test_row_sums_equal_one(self, encoded_labels):
        """
        Every row in the one-hot array must sum to exactly 1.0.
        Failure here means the one-hot conversion is corrupted.
        """
        _, y_encoded, _ = encoded_labels
        row_sums = y_encoded.sum(axis=1)
        assert np.allclose(row_sums, 1.0), (
            "One-hot rows should each sum to 1.0. "
            "Found rows summing to: " + str(row_sums[~np.isclose(row_sums, 1.0)])
        )

    def test_all_three_classes_encoded(self, encoded_labels):
        """All three risk classes must appear in the encoded dataset."""
        _, y_encoded, _ = encoded_labels
        classes_present = set(np.unique(np.argmax(y_encoded, axis=1)))
        assert classes_present == {0, 1, 2}, (
            f"Not all classes encoded. Found: {classes_present}"
        )

    def test_label_encoder_class_order(self, encoded_labels):
        """
        Label encoder must map Low=0, Medium=1, High=2.
        Wrong order would silently mislabel the confusion matrix.
        """
        _, _, le = encoded_labels
        assert list(le.classes_) == CLASS_ORDER, (
            f"Expected class order {CLASS_ORDER}, got {list(le.classes_)}"
        )



class TestDataSplitting:

    def test_train_size_correct(self, split_arrays):
        """Training set should contain 800 samples (80% of 1,000)."""
        X_train, *_ = split_arrays
        assert X_train.shape[0] == 800, (
            f"Expected 800 training samples, got {X_train.shape[0]}."
        )

    def test_val_size_correct(self, split_arrays):
        """Validation set should contain 100 samples (10% of 1,000)."""
        _, X_val, *_ = split_arrays
        assert X_val.shape[0] == 100, (
            f"Expected 100 validation samples, got {X_val.shape[0]}."
        )

    def test_test_size_correct(self, split_arrays):
        """Test set should contain 100 samples (10% of 1,000)."""
        _, _, X_test, *_ = split_arrays
        assert X_test.shape[0] == 100, (
            f"Expected 100 test samples, got {X_test.shape[0]}."
        )

    def test_total_samples_conserved(self, split_arrays):
        """Train + val + test must sum to 1,000 so no data lost or duplicated."""
        X_train, X_val, X_test, *_ = split_arrays
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == 1000, (
            f"Total samples after split = {total}, expected 1000. "
            f"Data may have been lost or duplicated."
        )

    def test_feature_count_preserved_after_split(self, split_arrays):
        """All splits must have 23 feature columns."""
        X_train, X_val, X_test, *_ = split_arrays
        for name, arr in [('train', X_train), ('val', X_val), ('test', X_test)]:
            assert arr.shape[1] == 23, (
                f"{name} set has {arr.shape[1]} features, expected 23."
            )



class TestNormalization:

    def test_train_mean_near_zero(self, scaled_arrays):
        """
        After StandardScaler, the per-column mean of the training
        set should be approximately 0 (within ±0.1).
        """
        X_train_s, *_ = scaled_arrays
        max_abs_mean = np.abs(X_train_s.mean(axis=0)).max()
        assert max_abs_mean < 0.1, (
            f"Max absolute column mean after scaling = {max_abs_mean:.4f}. "
            f"Expected < 0.1. Check that scaler was fit on training data."
        )

    def test_train_std_near_one(self, scaled_arrays):
        """
        After StandardScaler, the per-column std of the training
        set should be approximately 1 (within ±0.1).
        """
        X_train_s, *_ = scaled_arrays
        max_std_diff = np.abs(X_train_s.std(axis=0) - 1).max()
        assert max_std_diff < 0.1, (
            f"Max |std - 1| after scaling = {max_std_diff:.4f}. "
            f"Expected < 0.1."
        )

    def test_scaled_shape_matches_input(self, scaled_arrays, split_arrays):
        """Scaling must not change the shape of any split."""
        X_train_s, X_val_s, X_test_s, *_ = scaled_arrays
        X_train,   X_val,   X_test,   *_ = split_arrays
        assert X_train_s.shape == X_train.shape, "Train shape changed after scaling."
        assert X_val_s.shape   == X_val.shape,   "Val shape changed after scaling."
        assert X_test_s.shape  == X_test.shape,  "Test shape changed after scaling."

    def test_no_nan_after_scaling(self, scaled_arrays):
        """Scaling should not introduce NaN values."""
        X_train_s, X_val_s, X_test_s, *_ = scaled_arrays
        for name, arr in [('train', X_train_s), ('val', X_val_s), ('test', X_test_s)]:
            assert not np.isnan(arr).any(), (
                f"NaN values found in {name} after scaling."
            )



class TestModelArchitecture:

    def test_model_builds_without_error(self):
        """build_model() should return a compiled Keras model without raising."""
        model = build_model(input_dim=23)
        assert model is not None

    def test_output_layer_has_three_neurons(self):
        """Output layer must have 3 neurons so it's one per risk class."""
        model = build_model(input_dim=23)
        assert model.output_shape == (None, 3), (
            f"Output shape {model.output_shape}. expected (None, 3)."
        )

    def test_model_accepts_23_features(self):
        """Model must accept input tensors of shape (batch, 23)."""
        model = build_model(input_dim=23)
        dummy = np.zeros((4, 23))
        output = model.predict(dummy, verbose=0)
        assert output.shape == (4, 3), (
            f"Model output shape {output.shape} . expected (4, 3)."
        )

    def test_softmax_probabilities_sum_to_one(self):
        """
        Softmax output must sum to 1.0 per sample (within float tolerance).
        Failure here indicates the output activation is not softmax.
        """
        model = build_model(input_dim=23)
        dummy = np.random.rand(20, 23)
        proba = model.predict(dummy, verbose=0)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            "Softmax probabilities do not sum to 1.0. "
            "Check the output layer activation."
        )

    def test_all_probabilities_non_negative(self):
        """All probability values must be ≥ 0 (softmax property)."""
        model = build_model(input_dim=23)
        dummy = np.random.rand(10, 23)
        proba = model.predict(dummy, verbose=0)
        assert (proba >= 0).all(), "Negative probability values found."

    def test_model_has_optimizer(self):
        """Model must be compiled with an optimizer (Adam)."""
        model = build_model(input_dim=23)
        assert model.optimizer is not None

    def test_model_has_loss_function(self):
        """Model must have a loss function configured."""
        model = build_model(input_dim=23)
        assert model.loss is not None



class TestPredictionsAndEvaluation:

    def test_predictions_are_valid_class_indices(self, small_trained_model, scaled_arrays):
        """Every prediction must be 0 (Low), 1 (Medium), or 2 (High)."""
        X_train_s, _, X_test_s, *_ = scaled_arrays
        _, y_pred = get_predictions(small_trained_model, X_test_s)
        invalid = set(y_pred) - {0, 1, 2}
        assert len(invalid) == 0, (
            f"Found invalid class indices in predictions: {invalid}"
        )

    def test_prediction_count_matches_input(self, small_trained_model, scaled_arrays):
        """Number of predictions must equal number of test samples."""
        _, _, X_test_s, *_ = scaled_arrays
        _, y_pred = get_predictions(small_trained_model, X_test_s)
        assert len(y_pred) == X_test_s.shape[0], (
            f"Got {len(y_pred)} predictions for {X_test_s.shape[0]} samples."
        )

    def test_evaluate_model_returns_all_required_metrics(self, small_trained_model, scaled_arrays):
        """evaluate_model() must return accuracy, precision, recall, and f1."""
        _, _, X_test_s, _, _, y_test, _ = scaled_arrays
        results = evaluate_model(small_trained_model, X_test_s, y_test)
        for key in ['accuracy', 'precision', 'recall', 'f1']:
            assert key in results, f"Missing metric '{key}' in evaluation results."

    def test_accuracy_is_between_zero_and_one(self, small_trained_model, scaled_arrays):
        """Accuracy must be a valid probability (0.0 to 1.0)."""
        _, _, X_test_s, _, _, y_test, _ = scaled_arrays
        results = evaluate_model(small_trained_model, X_test_s, y_test)
        assert 0.0 <= results['accuracy'] <= 1.0, (
            f"Accuracy {results['accuracy']} is outside [0, 1]."
        )

    def test_per_class_analysis_returns_three_rows(self, small_trained_model, scaled_arrays):
        """
        analyze_by_risk_level() must return one row per class (3 rows).
        Fewer rows would mean a class was absent from the test set.
        """
        _, _, X_test_s, _, _, y_test, _ = scaled_arrays
        results  = evaluate_model(small_trained_model, X_test_s, y_test)
        risk_df  = analyze_by_risk_level(results['y_true'], results['y_pred'])
        assert risk_df.shape[0] == 3, (
            f"Expected 3 rows in per-class breakdown, got {risk_df.shape[0]}."
        )

    def test_per_class_total_sums_to_test_size(self, small_trained_model, scaled_arrays):
        """
        Sum of 'True Positives in Test' across all classes must equal 100
        (the number of test samples). Missing patients would indicate a bug.
        """
        _, _, X_test_s, _, _, y_test, _ = scaled_arrays
        results = evaluate_model(small_trained_model, X_test_s, y_test)
        risk_df = analyze_by_risk_level(results['y_true'], results['y_pred'])
        total   = risk_df['True Positives in Test'].sum()
        assert total == 100, (
            f"Per-class totals sum to {total}, expected 100."
        )