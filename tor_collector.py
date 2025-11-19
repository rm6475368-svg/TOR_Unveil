import requests
import pandas as pd
from datetime import datetime
import json

class TORDataCollector:
    def __init__(self):
        self.tor_list_url = "https://www.dan.me.uk/torlist/?exit"
        self.consensus_url = "https://collector.torproject.org/recent/relay-descriptors/consensuses/"
    
    def fetch_exit_nodes(self):
        """Fetch current TOR exit nodes"""
        try:
            response = requests.get(self.tor_list_url, timeout=10)
            exit_nodes = response.text.strip().split('\n')
            
            nodes_data = []
            for ip in exit_nodes[:50]:  # Limit to 50 for demo
                if ip.strip():
                    nodes_data.append({
                        'ip': ip.strip(),
                        'type': 'exit',
                        'timestamp': datetime.now(),
                        'country': self.get_country(ip.strip())
                    })
            
            return pd.DataFrame(nodes_data)
        except Exception as e:
            print(f"Error fetching exit nodes: {e}")
            return self.generate_sample_data()
    
    def get_country(self, ip):
        """Simulate geolocation (use ip-api.com for real data)"""
        countries = ['US', 'DE', 'NL', 'FR', 'SE', 'UK', 'CA']
        import random
        return random.choice(countries)
    
    def generate_sample_data(self):
        """Generate realistic sample TOR node data"""
        import random
        
        nodes = []
        for i in range(50):
            nodes.append({
                'ip': f"185.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                'type': random.choice(['entry', 'exit']),
                'timestamp': datetime.now(),
                'country': random.choice(['US', 'DE', 'NL', 'FR', 'SE'])
            })
        
        return pd.DataFrame(nodes)
    
    def generate_traffic_patterns(self, num_sessions=20):
        """Generate simulated traffic correlation data"""
        import random
        import numpy as np
        
        sessions = []
        for i in range(num_sessions):
            # Create correlated entry-exit patterns
            base_time = datetime.now().timestamp() - random.randint(0, 3600)
            packet_count = random.randint(50, 500)
            
            # Entry node traffic
            entry_pattern = np.random.exponential(0.1, packet_count)
            
            # Exit node traffic (correlated with slight delay)
            delay = random.uniform(0.2, 0.5)
            exit_pattern = entry_pattern + delay + np.random.normal(0, 0.05, packet_count)
            
            sessions.append({
                'session_id': f"SID_{i:04d}",
                'entry_node': f"185.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                'exit_node': f"192.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                'entry_pattern': entry_pattern.tolist(),
                'exit_pattern': exit_pattern.tolist(),
                'correlation_score': random.uniform(0.65, 0.98),
                'timestamp': base_time,
                'packet_size_avg': random.randint(500, 1500),
                'confidence': random.uniform(0.7, 0.95)
            })
        
        return pd.DataFrame(sessions)

# Test the collector
if __name__ == "__main__":
    collector = TORDataCollector()
    nodes = collector.fetch_exit_nodes()
    print(nodes.head())
    
    traffic = collector.generate_traffic_patterns()
    print(traffic.head())
