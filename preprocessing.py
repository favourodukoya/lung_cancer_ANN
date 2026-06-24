"""
This local module is for all data loading,feature selection, auditing, encoding, and
normalisation logic for the Cancer Risk ANN pipeline.
"""

#Import libraries
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


#Configuration
ADMIN_COLS = ['index', 'Patient Id'] # These columns carry no predictive signal

TARGET_COL = 'Level' #Target column

"""
Fixing the class order ensures consistent one-hot encoding every
time (i.e Low=0, Medium=1, High=2). Without this, scikit-learn library may assign 
alphabetical order (High=0, Low=1, Medium=2), which would make confusion matrix rows and columns harder to interpret.
"""
CLASS_ORDER = ['Low', 'Medium', 'High']

# Feature semantics map (From Dataset)
FEATURE_SEMANTICS = {
    'Age':                     'Non-modifiable risk factor; cancer incidence rises steeply with age.',
    'Gender':                  'Known sex-based differences in lung cancer incidence (1=Male, 2=Female).',
    'Air Pollution':           'Environmental carcinogen exposure.',
    'Alcohol use':             'Alcohol increases carcinogen susceptibility and immune suppression.',
    'Dust Allergy':            'Indicator of respiratory sensitivity to particulate matter.',
    'OccuPational Hazards':    'Workplace carcinogen exposure (asbestos, silica, diesel fumes etc.).',
    'Genetic Risk':            'Family history and hereditary predisposition score.',
    'chronic Lung Disease':    'Pre-existing lung disease reduces respiratory reserve and masks symptoms.',
    'Balanced Diet':           'Antioxidant intake is inversely linked to cancer risk.',
    'Obesity':                 'Obesity elevates systemic inflammation and cancer biomarkers.',
    'Smoking':                 'The single strongest modifiable risk factor for lung cancer.',
    'Passive Smoker':          'Second-hand smoke carries similar carcinogenic compounds.',
    'Chest Pain':              'Presenting symptom, pleuritic or tumour-related chest pain.',
    'Coughing of Blood':       'Haemoptysis is a red-flag symptom for lung malignancy.',
    'Fatigue':                 'Constitutional symptom common in systemic cancer.',
    'Weight Loss':             'Unexplained weight loss is a classic paraneoplastic sign.',
    'Shortness of Breath':     'Dyspnoea from tumour mass effect or pleural effusion.',
    'Wheezing':                'Caused by partial airway obstruction from a tumour.',
    'Swallowing Difficulty':   'Dysphagia suggests mediastinal involvement or oesophageal compression.',
    'Clubbing of Finger Nails':'Digital clubbing is a paraneoplastic sign of lung cancer.',
    'Frequent Cold':           'Immune compromise increases susceptibility to respiratory infections.',
    'Dry Cough':               'Persistent dry cough is one of the earliest lung cancer symptoms.',
    'Snoring':                 'Indicator of upper-airway compromise; correlated with sleep apnoea risk.',
}



#Functions
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV from disk and return it as a DataFrame.

    raise a FileNotFoundError immediately if the path is wrong.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        Raw DataFrame with all original columns intact.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
        )

    df = pd.read_csv(filepath)
    print(f"[Load] Loaded '{filepath}' → {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Audit the dataset for missing values and report the result.

    The dataset has zero missing values on inspection, but I run this
    check on every pipeline execution. If the source CSV is ever updated
    or swapped with a different version, the pipeline catches problems
    immediately rather than silently passing NaNs into normalization.

    Args:
        df: The full raw DataFrame.

    Returns:
        Series with missing-value counts per column (all zeros here).
    """
    missing = df.isnull().sum()
    total   = missing.sum()

    if total == 0:
        print(f"[Missing Values] None found across all {df.shape[1]} columns. ✓")
    else:
        print(f"[Missing Values] ⚠ {total} missing values detected:")
        print(missing[missing > 0].to_string())

    return missing


def check_outliers(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Identify statistical outliers in each feature using the IQR method.

    Args:
        df: Raw DataFrame (used for context on column scales).
        feature_cols: List of feature column names to audit.

    Returns:
        DataFrame with Q1, Q3, IQR, fences, and outlier count per feature.
    """
    records = []
    for col in feature_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        records.append({
            'Feature':        col,
            'Q1':             Q1,
            'Q3':             Q3,
            'IQR':            IQR,
            'Lower Fence':    round(lower, 2),
            'Upper Fence':    round(upper, 2),
            'IQR Outliers':   n_out,
            'Note':           'Retained, valid ordinal extreme' if n_out > 0 else 'None'
        })

    report = pd.DataFrame(records)
    total_flagged = report['IQR Outliers'].sum()
    print(f"[Outliers] IQR analysis complete. {total_flagged} statistical outliers flagged across all features.")
    
    return report


def select_features(df: pd.DataFrame) -> tuple:
    """
    Separate the feature matrix (X) from the target label (y) and
    drop all administrative columns.

    Args:
        df: Full raw DataFrame.

    Returns:
        Tuple (X: feature DataFrame with 23 columns,
               y: label Series with 'Low'/'Medium'/'High' values).
    """
    # Drop admin columns, then separate features from target
    df_clean = df.drop(columns=ADMIN_COLS, errors='ignore')
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    print(f"[Features] Kept {X.shape[1]} features after dropping admin columns.")
    print(f"[Target]   Distribution: {y.value_counts().to_dict()}")
    return X, y


def encode_labels(y: pd.Series) -> tuple:
    """
    Convert string labels ('Low', 'Medium', 'High') into one-hot arrays
    suitable for a softmax output layer with categorical crossentropy loss.

    Args:
        y: Series of string class labels.

    Returns:
        Tuple (y_one_hot: numpy array of shape (n, 3),
               label_encoder: fitted LabelEncoder for decoding predictions).
    """
    le = LabelEncoder()
    # Manually set class order rather than fitting as this guarantees
    # the encoding is always Low=0, Medium=1, High=2 regardless of the
    # alphabetical ordering sklearn would otherwise use.
    le.classes_ = np.array(CLASS_ORDER)
    y_int     = le.transform(y)
    y_one_hot = to_categorical(y_int, num_classes=len(CLASS_ORDER))

    print(f"[Encoding] One-hot encoded. Mapping: {dict(zip(CLASS_ORDER, range(3)))}")
    return y_one_hot, le


def split_data(X: pd.DataFrame,
               y: np.ndarray,
               test_size:  float = 0.10,
               val_size:   float = 0.10,
               random_state: int = 42) -> tuple:
    """
    Split data into train / validation / test sets (80 / 10 / 10).

    Args:
        X: Feature DataFrame.
        y: One-hot encoded label array.
        test_size:     Fraction to reserve for testing (0.10 = 100 samples).
        val_size:      Fraction to reserve for validation (0.10 = 100 samples).
        random_state:  Random seed for reproducibility.

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Stratify on integer labels (argmax of one-hot) so the class
    # distribution is preserved across all three splits.
    y_int = np.argmax(y, axis=1)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_int
    )

    # Recalculate val_size relative to the remaining (non-test) data
    val_relative = val_size / (1.0 - test_size)
    y_temp_int   = np.argmax(y_temp, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_temp_int
    )

    print(f"[Split] Train={X_train.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train: np.ndarray,
                        X_val:   np.ndarray,
                        X_test:  np.ndarray,
                        scaler_save_path: str = 'scaler.pkl') -> tuple:
    """
    Standardize all features to zero mean and unit variance.

    Args:
        X_train, X_val, X_test: Split feature arrays (numpy).
        scaler_save_path: File path to save the fitted scaler.

    Returns:
        Tuple (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s  = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    joblib.dump(scaler, scaler_save_path)
    print(f"[Normalization] StandardScaler applied. Scaler saved → '{scaler_save_path}'.")
    return X_train_s, X_val_s, X_test_s, scaler


def run_preprocessing(filepath: str,
                       scaler_save_path: str = 'scaler.pkl') -> dict:
    """
    Full preprocessing pipeline

    Runs every preprocessing step in the correct order and returns
    all outputs as a dictionary so main.py and the test suite can
    unpack only what they need.

    Args:
        filepath: Path to the raw CSV file.
        scaler_save_path: Where to save the fitted StandardScaler.

    Returns:
        Dictionary containing:
          X_train, X_val, X_test  - scaled numpy arrays
          y_train, y_val, y_test  - one-hot label arrays
          label_encoder           - fitted LabelEncoder (for decoding)
          scaler                  - fitted StandardScaler (for app)
          feature_names           - list of 23 feature column names
          outlier_report          - DataFrame from check_outliers()
    """
    print("\n" + "─" * 55)
    print("  PREPROCESSING PIPELINE")
    print("─" * 55)

    df = load_data(filepath)
    check_missing_values(df)

    X, y = select_features(df)
    outlier_report = check_outliers(df, X.columns.tolist())

    y_encoded, le = encode_labels(y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded)

    X_train_s, X_val_s, X_test_s, scaler = normalize_features(
        X_train.values, X_val.values, X_test.values,
        scaler_save_path=scaler_save_path
    )

    print("─" * 55)
    print("  Preprocessing complete.\n")

    return {
        'X_train':        X_train_s,
        'X_val':          X_val_s,
        'X_test':         X_test_s,
        'y_train':        y_train,
        'y_val':          y_val,
        'y_test':         y_test,
        'label_encoder':  le,
        'scaler':         scaler,
        'feature_names':  X.columns.tolist(),
        'outlier_report': outlier_report,
    }