"""Walk-forward cross-validation for time-series ML models.

Implements sliding window CV with purge gap to prevent data leakage.
Standard k-fold CV is INVALID for financial time series.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """A single train/test split."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class WalkForwardCV:
    """
    Sliding window walk-forward cross-validation with purge gap.

    Parameters:
    - train_size: number of bars in training window
    - test_size: number of bars in test window
    - purge_gap: number of bars to skip between train and test (prevents leakage)
    - step_size: how many bars to slide forward per fold (default = test_size)
    - expanding: if True, training window grows (walk-forward); if False, fixed window (sliding)
    """

    def __init__(
        self,
        train_size: int = 500,
        test_size: int = 100,
        purge_gap: int = 5,
        step_size: int | None = None,
        expanding: bool = False,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.step_size = step_size or test_size
        self.expanding = expanding

    def split(self, n_samples: int) -> list[WalkForwardFold]:
        """
        Generate train/test fold indices.

        Returns list of WalkForwardFold with integer indices into the data.
        """
        folds = []
        fold_idx = 0

        if self.expanding:
            # Expanding window: train always starts at 0
            train_start = 0
            test_start = self.train_size + self.purge_gap

            while test_start + self.test_size <= n_samples:
                fold = WalkForwardFold(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=test_start - self.purge_gap,
                    test_start=test_start,
                    test_end=test_start + self.test_size,
                )
                folds.append(fold)
                fold_idx += 1
                test_start += self.step_size
        else:
            # Sliding window: fixed train size
            train_start = 0

            while True:
                train_end = train_start + self.train_size
                test_start = train_end + self.purge_gap
                test_end = test_start + self.test_size

                if test_end > n_samples:
                    break

                fold = WalkForwardFold(
                    fold_idx=fold_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
                folds.append(fold)
                fold_idx += 1
                train_start += self.step_size

        logger.info(
            "Walk-forward CV: %d folds, train=%d, test=%d, purge=%d, %s window",
            len(folds), self.train_size, self.test_size, self.purge_gap,
            "expanding" if self.expanding else "sliding",
        )
        return folds

    def split_dataframe(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Split X and y DataFrames into (X_train, y_train, X_test, y_test) tuples.
        """
        folds = self.split(len(X))
        result = []

        for fold in folds:
            X_train = X.iloc[fold.train_start:fold.train_end]
            y_train = y.iloc[fold.train_start:fold.train_end]
            X_test = X.iloc[fold.test_start:fold.test_end]
            y_test = y.iloc[fold.test_start:fold.test_end]

            # Drop any NaN labels and inf feature values
            train_mask = y_train.notna() & np.isfinite(X_train).all(axis=1)
            test_mask = y_test.notna() & np.isfinite(X_test).all(axis=1)

            result.append((
                X_train[train_mask], y_train[train_mask],
                X_test[test_mask], y_test[test_mask],
            ))

        return result

    def get_holdout_split(
        self, X: pd.DataFrame, y: pd.Series, holdout_pct: float = 0.15
    ) -> tuple:
        """
        Split off a final holdout test set that is NEVER used during CV.

        Returns: (X_cv, y_cv, X_holdout, y_holdout)
        """
        n = len(X)
        holdout_size = int(n * holdout_pct)
        split_idx = n - holdout_size

        X_cv = X.iloc[:split_idx]
        y_cv = y.iloc[:split_idx]
        X_holdout = X.iloc[split_idx:]
        y_holdout = y.iloc[split_idx:]

        logger.info("Holdout split: CV=%d samples, Holdout=%d samples (%.0f%%)",
                     len(X_cv), len(X_holdout), holdout_pct * 100)
        return X_cv, y_cv, X_holdout, y_holdout


def auto_configure_cv(n_samples: int, min_folds: int = 3) -> WalkForwardCV:
    """
    Auto-configure walk-forward CV based on dataset size.

    Heuristic:
    - Train size: ~60% of data / min_folds
    - Test size: ~15% of data / min_folds
    - Purge: 5 bars
    """
    # Reserve 15% for holdout, use remaining 85% for CV
    cv_samples = int(n_samples * 0.85)

    # Each fold needs train + purge + test
    test_size = max(20, cv_samples // (min_folds * 5))
    train_size = max(50, cv_samples // min_folds - test_size - 5)
    purge_gap = 5

    cv = WalkForwardCV(
        train_size=train_size,
        test_size=test_size,
        purge_gap=purge_gap,
        expanding=False,
    )

    actual_folds = len(cv.split(cv_samples))
    if actual_folds < min_folds:
        # Shrink windows to get enough folds
        total_per_fold = cv_samples // min_folds
        test_size = max(15, total_per_fold // 5)
        train_size = total_per_fold - test_size - purge_gap
        cv = WalkForwardCV(train_size=train_size, test_size=test_size, purge_gap=purge_gap)

    logger.info("Auto-configured CV for %d samples: train=%d, test=%d, purge=%d",
                n_samples, cv.train_size, cv.test_size, cv.purge_gap)
    return cv
