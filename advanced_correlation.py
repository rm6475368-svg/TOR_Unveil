import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from correlation_engine import CorrelationEngine
import pickle

class AdvancedCorrelationEngine(CorrelationEngine):
    """ML + Statistics Enhanced Correlation"""
    
    def __init__(self):
        super().__init__()
        self.ml_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'Entry Mean', 'Entry Std', 'Entry Median', 'Entry Range', 'Entry Kurtosis',
            'Exit Mean', 'Exit Std', 'Correlation', 'Mean Diff', 'Std Diff',
            'Entropy Entry', 'Entropy Exit', 'Entropy Diff',
            'Entry Autocorr', 'Exit Autocorr', 'FFT Correlation',
            'Min Value', 'Max Value', 'Skewness', 'Outlier Ratio'
        ]
    
    def extract_features(self, entry_pattern, exit_pattern):
        """Extract 20 ML features"""
        features = []
        
        # Entry pattern statistics
        features.append(np.mean(entry_pattern))
        features.append(np.std(entry_pattern))
        features.append(np.median(entry_pattern))
        features.append(np.max(entry_pattern) - np.min(entry_pattern))
        
        try:
            features.append(stats.kurtosis(entry_pattern))
        except:
            features.append(0)
        
        # Exit pattern statistics
        features.append(np.mean(exit_pattern))
        features.append(np.std(exit_pattern))
        
        # Correlation
        corr = np.corrcoef(entry_pattern, exit_pattern)[0, 1]
        features.append(np.nan_to_num(corr, 0))
        features.append(abs(np.mean(entry_pattern) - np.mean(exit_pattern)))
        features.append(abs(np.std(entry_pattern) - np.std(exit_pattern)))
        
        # Entropy
        features.append(self.calculate_entropy(entry_pattern))
        features.append(self.calculate_entropy(exit_pattern))
        features.append(abs(
            self.calculate_entropy(entry_pattern) - 
            self.calculate_entropy(exit_pattern)
        ))
        
        # Autocorrelation
        features.append(self.calculate_autocorr(entry_pattern))
        features.append(self.calculate_autocorr(exit_pattern))
        
        # FFT correlation
        try:
            fft_corr = self.calculate_fft_correlation(entry_pattern, exit_pattern)
            features.append(fft_corr)
        except:
            features.append(0)
        
        # Min/Max
        features.append(np.min(entry_pattern))
        features.append(np.max(entry_pattern))
        
        # Skewness
        try:
            features.append(stats.skew(entry_pattern))
        except:
            features.append(0)
        
        # Outlier ratio
        q1 = np.percentile(entry_pattern, 25)
        q3 = np.percentile(entry_pattern, 75)
        iqr = q3 - q1
        outliers = np.sum((entry_pattern < q1 - 1.5*iqr) | (entry_pattern > q3 + 1.5*iqr))
        features.append(outliers / len(entry_pattern) if len(entry_pattern) > 0 else 0)
        
        return np.array(features)
    
    def calculate_entropy(self, data):
        """Shannon entropy"""
        if len(data) < 2:
            return 0
        
        normalized = ((data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10) * 255).astype(int)
        hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0
        
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob + 1e-10))
    
    def calculate_autocorr(self, data):
        """Autocorrelation at lag 1"""
        if len(data) < 2:
            return 0
        
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / len(data)
        c1 = np.sum((data[:-1] - mean) * (data[1:] - mean)) / len(data)
        
        return c1 / c0 if c0 != 0 else 0
    
    def calculate_fft_correlation(self, entry, exit):
        """Frequency domain correlation"""
        fft_entry = np.abs(np.fft.fft(entry))
        fft_exit = np.abs(np.fft.fft(exit))
        
        return np.corrcoef(fft_entry, fft_exit)[0, 1]
    
    def train_model(self, training_pairs, labels):
        """Train on labeled data"""
        X = []
        for entry, exit in training_pairs:
            features = self.extract_features(entry, exit)
            X.append(features)
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        self.ml_model.fit(X_scaled, labels)
        self.is_trained = True
        
        accuracy = self.ml_model.score(X_scaled, labels)
        return accuracy
    
    def predict_correlation_ml(self, entry_pattern, exit_pattern):
        """ML-based prediction"""
        if not self.is_trained:
            return self.time_based_correlation(entry_pattern, exit_pattern)
        
        features = self.extract_features(entry_pattern, exit_pattern)
        features_scaled = self.scaler.transform([features])
        
        probability = self.ml_model.predict_proba(features_scaled)
        return probability
    
    def get_feature_importance(self):
        """Which features matter most"""
        if not self.is_trained:
            return None
        
        importance = self.ml_model.feature_importances_
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    
    def calculate_confidence_interval(self, correlation, sample_size, confidence=0.95):
        """95% confidence interval using Fisher transform"""
        if abs(correlation) >= 0.99:
            return correlation, correlation
        
        z = np.arctanh(np.clip(correlation, -0.99, 0.99))
        se = 1 / np.sqrt(sample_size - 3)
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        
        ci_lower = np.tanh(z - z_critical * se)
        ci_upper = np.tanh(z + z_critical * se)
        
        return ci_lower, ci_upper
    
    def hypothesis_test(self, entry_pattern, exit_pattern):
        """Rigorous statistical test"""
        correlation = np.corrcoef(entry_pattern, exit_pattern)[0, 1]
        n = len(entry_pattern)
        
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2 + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        cohens_d = 2 * t_stat / np.sqrt(n)
        
        return {
            'correlation': correlation,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    
    def save_model(self, filepath):
        """Save for production"""
        pickle.dump({
            'model': self.ml_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }, open(filepath, 'wb'))
    
    def load_model(self, filepath):
        """Load pre-trained"""
        data = pickle.load(open(filepath, 'rb'))
        self.ml_model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
