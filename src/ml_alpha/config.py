# All configurable constants for the ML alpha research system.
# Change values here only — never use magic numbers elsewhere in the codebase.
# Every constant has a comment explaining what it controls and why this value was chosen.

# ── Data ──────────────────────────────────────────────────────────────────────

TICKER: str = "SPY"
# S&P 500 ETF. Liquid, clean adjusted price history back to 1993.
# Consistent with the backtesting engine so both projects operate on the same instrument.

START_DATE: str = "1993-01-01"
# SPY inception date. Maximises available history for walk-forward windows.

ANNUALISATION_FACTOR: int = 252
# Trading days per year. Used to annualise daily return statistics.

# ── Walk-forward windows ───────────────────────────────────────────────────────

TRAINING_WINDOW_YEARS: int = 3
# Length of each rolling training window in years (~756 trading days).
# Three years captures at least one full bull/bear cycle without anchoring the model
# to conditions that are too distant to be relevant to the test period.
# Matches the backtesting engine's window length for direct comparability.

TESTING_WINDOW_YEARS: int = 1
# Length of each out-of-sample test window in years (~252 trading days).
# One year provides a meaningful evaluation period while keeping training data recent.
# The 3:1 train/test ratio is standard in walk-forward validation for daily financial data.

# ── Feature construction ───────────────────────────────────────────────────────

MOMENTUM_WINDOWS: list[int] = [5, 10, 20, 60]
# Lookback windows (in trading days) for rolling return features.
# 5 days ~ one week (short-term reversal horizon).
# 10 days ~ two weeks (transitional).
# 20 days ~ one month (medium momentum).
# 60 days ~ one quarter (intermediate momentum, documented in academic literature).

VOLATILITY_WINDOWS: list[int] = [10, 20]
# Lookback windows for rolling standard deviation of returns.
# Captures volatility clustering at two horizons: short (10-day) and medium (20-day).

VOLUME_WINDOW: int = 20
# Lookback window for rolling average volume used in the volume ratio feature.
# 20 days ~ one trading month. Standard reference period for volume normalisation.

RSI_WINDOW: int = 14
# Lookback window for the Relative Strength Index.
# 14 days is the standard RSI period introduced by Welles Wilder in 1978
# and remains the most widely used setting in practice.

BOLLINGER_WINDOW: int = 20
BOLLINGER_STD: float = 2.0
# Bollinger Band parameters: 20-day rolling mean ± 2 standard deviations.
# The 20/2 combination is the original specification by John Bollinger
# and is the de facto industry standard.

MIN_FEATURE_ROWS: int = 60
# Minimum number of rows required after feature construction before any NaN rows
# are dropped. The longest lookback window (60 days for momentum) means the first
# 60 rows will have NaN features. This constant is used in validation to ensure
# enough data remains after dropping those rows.

# ── Model ─────────────────────────────────────────────────────────────────────

RANDOM_STATE: int = 42
# Fixed random seed for all stochastic operations (model training, any sampling).
# Ensures reproducibility: the same data always produces the same model.

N_ESTIMATORS: int = 200
# Number of boosting rounds (trees) in the gradient boosting model.
# 200 is a reasonable default — large enough to learn complex patterns,
# small enough to train quickly and avoid severe overfitting.

MAX_DEPTH: int = 3
# Maximum depth of each decision tree in the ensemble.
# Shallow trees (depth 3-4) are appropriate for noisy financial data:
# deep trees overfit the noise rather than learning the signal.

LEARNING_RATE: float = 0.05
# Shrinkage applied to each tree's contribution.
# Smaller values require more trees but generalise better.
# 0.05 with 200 estimators is a standard conservative combination.

SUBSAMPLE: float = 0.8
# Fraction of training rows sampled (without replacement) to fit each tree.
# Introduces randomness that reduces overfitting (stochastic gradient boosting).

# ── Signal generation ─────────────────────────────────────────────────────────

BUY_THRESHOLD: float = 0.0005
# Minimum predicted return to generate a buy signal (0.05% predicted gain).
# Set above zero to avoid trading on predictions that are too small to overcome
# transaction costs. This value is a starting point — it should be tuned
# empirically on the training window.

SELL_THRESHOLD: float = -0.0005
# Maximum predicted return to generate a sell signal (-0.05% predicted loss).
# Symmetric with BUY_THRESHOLD. Predictions between the two thresholds produce
# a hold signal (0), keeping the strategy flat when conviction is low.

# ── Recency weighting ─────────────────────────────────────────────────────────

RECENCY_DECAY_RATE: float = 0.005
# Exponential decay rate for sample weights during training.
# weight_t = exp(RECENCY_DECAY_RATE * position_from_start)
# where position_from_start runs from 0 (oldest) to N-1 (most recent).
# At 0.005, an observation 200 days old receives weight exp(-1.0) ≈ 0.37
# relative to the most recent observation. Gives recent data more influence
# without completely discarding older observations.
# Set to 0.0 to disable recency weighting (all rows equally weighted).