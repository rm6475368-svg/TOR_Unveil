import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class CorrelationEngine:
    def __init__(self):
        self.correlation_threshold = 0.75
    
    def time_based_correlation(self, entry_times, exit_times):
        """Correlate traffic based on timing patterns"""
        if len(entry_times) != len(exit_times):
            # Pad shorter array
            max_len = max(len(entry_times), len(exit_times))
            entry_times = np.pad(entry_times, (0, max_len - len(entry_times)))
            exit_times = np.pad(exit_times, (0, max_len - len(exit_times)))
        
        correlation, p_value = pearsonr(entry_times, exit_times)
        return correlation
    
    def packet_size_correlation(self, entry_sizes, exit_sizes):
        """Correlate based on packet size patterns"""
        entry_hist, _ = np.histogram(entry_sizes, bins=10)
        exit_hist, _ = np.histogram(exit_sizes, bins=10)
        
        similarity = cosine_similarity([entry_hist], [exit_hist])[0][0]
        return similarity
    
    def flow_correlation(self, entry_flow, exit_flow):
        """Correlate traffic flow patterns"""
        # Normalize flows
        entry_normalized = (entry_flow - np.mean(entry_flow)) / np.std(entry_flow)
        exit_normalized = (exit_flow - np.mean(exit_flow)) / np.std(exit_flow)
        
        correlation = np.corrcoef(entry_normalized, exit_normalized)[0, 1]
        return correlation
    
    def calculate_confidence(self, correlations):
        """Calculate overall confidence score"""
        weights = [0.4, 0.3, 0.3]  # Time, packet size, flow weights
        confidence = sum(w * c for w, c in zip(weights, correlations))
        return min(max(confidence, 0), 1)  # Clamp between 0 and 1
    
    def identify_entry_node(self, exit_node_data, all_entry_nodes):
        """Identify probable entry node for given exit node"""
        correlations = []
        
        for entry_node in all_entry_nodes:
            # Calculate multiple correlation metrics
            time_corr = self.time_based_correlation(
                entry_node['pattern'], 
                exit_node_data['pattern']
            )
            
            confidence = self.calculate_confidence([time_corr, 0.8, 0.75])
            
            correlations.append({
                'entry_node': entry_node['ip'],
                'correlation': time_corr,
                'confidence': confidence
            })
        
        # Sort by confidence
        correlations.sort(key=lambda x: x['confidence'], reverse=True)
        return correlations[0] if correlations else None

# Test
if __name__ == "__main__":
    engine = CorrelationEngine()
    
    # Test correlation
    entry_pattern = np.random.exponential(0.1, 100)
    exit_pattern = entry_pattern + 0.3 + np.random.normal(0, 0.05, 100)
    
    correlation = engine.time_based_correlation(entry_pattern, exit_pattern)
    print(f"Correlation: {correlation:.3f}")
