import cudf
import cupy as cp

class GPUStandardScaler:
    """
    Implements StandardScaler for GPU using cuDF and CuPy.
    - Standardizes features to mean=0, std=1.
    - Works on cuDF DataFrames (GPU).
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, df: cudf.DataFrame):
        """
        Computes the mean and standard deviation for scaling.

        Args:
            df (cudf.DataFrame): Input dataset (features only, no labels).
        """
        self.mean_ = df.mean()  # Compute mean (cuDF GPU operation)
        self.std_ = df.std()  # Compute standard deviation (cuDF GPU operation)
        self.std_ = self.std_.replace(0, 1)  # Avoid division by zero

    def transform(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Applies standardization (Z-score normalization) to the dataset.

        Args:
            df (cudf.DataFrame): Input dataset to transform.

        Returns:
            cudf.DataFrame: Standardized dataset.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call `fit` first.")

        return (df - self.mean_) / self.std_

    def fit_transform(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Fits the scaler to the dataset and transforms it.

        Args:
            df (cudf.DataFrame): Input dataset.

        Returns:
            cudf.DataFrame: Standardized dataset.
        """
        self.fit(df)
        return self.transform(df)
