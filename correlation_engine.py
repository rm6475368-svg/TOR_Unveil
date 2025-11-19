# correlation_engine.py

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class CorrelationEngine:
    """
    Basic correlation engine with time-based correlation
    """
    def time_based_correlation(self, entry_pattern, exit_pattern):
        """
        Compute simple Pearson correlation between two packet timing arrays.
        """
        if len(entry_pattern) < 2 or len(exit_pattern) < 2:
            return 0.0
        min_len = min(len(entry_pattern), len(exit_pattern))
        return np.corrcoef(entry_pattern[:min_len], exit_pattern[:min_len])[0, 1]


class AdvancedCorrelationEngine(CorrelationEngine):
    """
    Correlation engine with ML scoring and statistical rigor
    """
    def __init__(self):
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, entry_pattern, exit_pattern):
        """
        Extract feature vector from traffic patterns for ML prediction.
        For demo, simple statistics of timestamps.
        """
        min_len = min(len(entry_pattern), len(exit_pattern))
        if min_len == 0:
            return np.zeros((1, 6))
        ep = np.array(entry_pattern[:min_len])
        xp = np.array(exit_pattern[:min_len])

        features = [
            np.corrcoef(ep, xp)[0, 1],          # correlation
            np.mean(ep),                        # entry mean time
            np.std(ep),                        # entry std dev
            np.mean(xp),                       # exit mean time
            np.std(xp),                       # exit std dev
            np.abs(np.mean(ep) - np.mean(xp))  # mean difference
        ]
        return np.array(features).reshape(1, -1)

    def train_model(self, training_data, labels):
        """
        Train ML model on given data.
        training_data: list of (entry_pattern, exit_pattern)
        labels: list of 0/1 indicating whether patterns match
        """
        X = []
        for ep, xp in training_data:
            X.append(self.extract_features(ep, xp).flatten())
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, labels)
        self.is_trained = True

    def predict_correlation(self, entry_pattern, exit_pattern):
        """
        Predict correlation (probability) using ML model if trained,
        else fall back to time-based correlation.
        """
        if not self.is_trained:
            return self.time_based_correlation(entry_pattern, exit_pattern)
        features = self.extract_features(entry_pattern, exit_pattern)
        features_scaled = self.scaler.transform(features)
        proba = self.ml_model.predict_proba(features_scaled)[0][1]
        return proba

    def calculate_confidence_interval(self, correlation, sample_size, confidence=0.95):
        """
        Calculate confidence interval for Pearson correlation using Fisher Z-transform.
        """
        if sample_size < 4:
            # Not enough samples to calculate a reliable interval
            return (correlation, correlation)
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(sample_size - 3)
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        lo_z, hi_z = z - z_critical * se, z + z_critical * se
        lo, hi = np.tanh(lo_z), np.tanh(hi_z)
        return (lo, hi)

    def hypothesis_test(self, entry_pattern, exit_pattern, alpha=0.05):
        """
        Perform hypothesis test on correlation coefficient.
        Null hypothesis H0: correlation = 0
        Return p-value and significance.
        """
        correlation = self.time_based_correlation(entry_pattern, exit_pattern)
        n = min(len(entry_pattern), len(exit_pattern))
        if n < 3:
            return {"p_value": 1.0, "significant": False, "effect_size": "undefined"}

        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation ** 2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
        significant = p_value < alpha

        # Effect size interpretation (Cohen's guidelines)
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            effect_size = "negligible"
        elif abs_corr < 0.3:
            effect_size = "small"
        elif abs_corr < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"

        return {"p_value": p_value, "significant": significant, "effect_size": effect_size}

# ---------------- Entry/Guard Node “True” Identification ----------------

def compute_correlation(p1, p2):
    """
    Simple Pearson correlation between two timing patterns.
    """
    if len(p1) < 2 or len(p2) < 2:
        return 0.0
    min_len = min(len(p1), len(p2))
    corr = np.corrcoef(np.array(p1[:min_len]), np.array(p2[:min_len]))[0, 1]
    if np.isnan(corr):
        return 0.0
    return corr

def correlate_entry_and_guard(edge_df, guard_df):
    """
    Correlate edge (user/network) sessions with guard node sessions.

    edge_df: DataFrame with columns ['session_id', 'entry_node', 'entry_pattern', ...]
    guard_df: DataFrame same as edge_df from guard node captures.

    Returns DataFrame with best matches and correlation scores.
    """
    results = []
    for i, user_row in edge_df.iterrows():
        best_match = None
        best_score = float("-inf")
        for j, guard_row in guard_df.iterrows():
            score = compute_correlation(user_row['entry_pattern'], guard_row['entry_pattern'])
            if score > best_score:
                best_score = score
                best_match = guard_row
        results.append({
            "user_session": user_row["session_id"],
            "user_ip": user_row["entry_node"],
            "matched_guard_session": best_match["session_id"] if best_match is not None else None,
            "guard_ip": best_match["entry_node"] if best_match is not None else None,
            "correlation_score": best_score
        })
    return pd.DataFrame(results)
