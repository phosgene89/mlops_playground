import numpy as np

def create_multistep_lagged_numpy(data, n_lags=10, n_steps=3):
    """
    data: 1D array-like (e.g., df["adj_close"].values)
    n_lags: number of lag steps to use as input
    n_steps: number of future steps to predict
    Returns: X (samples x n_lags), y (samples x n_steps)
    """
    data = np.asarray(data)
    n_samples = len(data) - n_lags - n_steps + 1

    X = np.array([data[i:i+n_lags] for i in range(n_samples)])
    y = np.array([data[i+n_lags:i+n_lags+n_steps] for i in range(n_samples)])

    return X, y

def create_multistep_lagged_numpy(data, n_lags=10, n_steps=3, as_direction=True):
    """
    data: 1D array-like (e.g., df["adj_close"].values)
    n_lags: number of lag steps to use as input
    n_steps: number of future steps to predict
    as_direction: if True, convert both X and y to binary up/down (1 = up, 0 = down)
    
    Returns:
        X: shape (samples, n_lags) — raw prices or directional movement
        y: shape (samples, n_steps) — future prices or binary up/down
    """
    data = np.asarray(data)
    n_samples = len(data) - n_lags - n_steps + 1

    # X: input features
    X_raw = np.array([data[i:i+n_lags] for i in range(n_samples)])

    if as_direction:
        # Convert inputs to binary up/down relative to previous step
        X = (X_raw[:, 1:] > X_raw[:, :-1]).astype(int)  # shape (n_samples, n_lags-1)
    else:
        X = X_raw

    # y: future targets
    if as_direction:
        y = np.array([
            (data[i+n_lags:i+n_lags+n_steps] > data[i+n_lags-1]).astype(int)
            for i in range(n_samples)
        ])
    else:
        y = np.array([
            data[i+n_lags:i+n_lags+n_steps]
            for i in range(n_samples)
        ])

    return X, y