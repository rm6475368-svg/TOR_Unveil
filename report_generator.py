import pandas as pd
from datetime import datetime
import json

class ForensicReportGenerator:
    def __init__(self):
        self.report_data = {}
    
    def generate_markdown_report(self, traffic_data, node_data, confidence_threshold=0.8):
        """Generate detailed forensic report"""
        
        high_confidence = traffic_data[traffic_data['confidence'] >= confidence_threshold]
        
        report = f"""
# TOR Network Forensic Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Period:** Last 24 hours  
**Confidence Threshold:** {confidence_threshold * 100}%

---

## Executive Summary

- **Total Sessions Analyzed:** {len(traffic_data)}
- **High Confidence Matches:** {len(high_confidence)} ({len(high_confidence)/len(traffic_data)*100:.1f}%)
- **Unique Entry Nodes:** {traffic_data['entry_node'].nunique()}
- **Unique Exit Nodes:** {traffic_data['exit_node'].nunique()}
- **Average Correlation Score:** {traffic_data['correlation_score'].mean():.3f}

---

## Identified Entry-Exit Node Pairs

"""
        
        for idx, row in high_confidence.iterrows():
            report += f"""
### Session ID: {row['session_id']}

- **Entry Node IP:** `{row['entry_node']}`
- **Exit Node IP:** `{row['exit_node']}`
- **Correlation Score:** {row['correlation_score']:.3f}
- **Confidence Level:** {row['confidence']*100:.1f}%
- **Timestamp:** {datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
- **Packet Count:** {len(row['entry_pattern'])}
- **Average Packet Size:** {row['packet_size_avg']} bytes

**Analysis:** Based on time-based correlation, packet size patterns, and flow characteristics, 
this entry-exit node pair shows {row['confidence']*100:.1f}% probability of being part of the same circuit.

---
"""
        
        report += f"""
## Node Activity Summary

### Top 5 Most Active Entry Nodes

"""
        entry_counts = traffic_data['entry_node'].value_counts().head(5)
        for ip, count in entry_counts.items():
            report += f"- `{ip}`: {count} connections\n"
        
        report += f"""
### Top 5 Most Active Exit Nodes

"""
        exit_counts = traffic_data['exit_node'].value_counts().head(5)
        for ip, count in exit_counts.items():
            report += f"- `{ip}`: {count} connections\n"
        
        report += """
---

## Methodology

### Data Collection
1. Automated extraction of TOR relay and node details from public directories
2. Real-time monitoring of entry and exit node traffic patterns

### Correlation Techniques
1. **Time-based Correlation:** Analyzing packet timing patterns between entry and exit nodes
2. **Packet Size Analysis:** Matching packet size distributions across nodes
3. **Flow Pattern Recognition:** Identifying unique traffic flow characteristics

### Confidence Scoring
Confidence scores are calculated using weighted combination of:
- Time correlation (40%)
- Packet size similarity (30%)
- Flow pattern matching (30%)

---

## Recommendations

1. **High Priority Investigation:** Focus on sessions with confidence > 90%
2. **Network Monitoring:** Deploy additional monitoring on identified entry nodes
3. **Pattern Analysis:** Continue long-term analysis for improved accuracy
4. **PCAP Integration:** Integrate actual PCAP data for enhanced correlation

---

## Legal Notice

This report is generated for law enforcement purposes under applicable legal frameworks.
All data collection and analysis methods comply with authorized investigation protocols.

**Report End**
"""
        
        return report
    
    def export_json(self, traffic_data, filename='forensic_export.json'):
        """Export data in JSON format"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'sessions': []
        }
        
        for _, row in traffic_data.iterrows():
            export_data['sessions'].append({
                'session_id': row['session_id'],
                'entry_node': row['entry_node'],
                'exit_node': row['exit_node'],
                'correlation_score': float(row['correlation_score']),
                'confidence': float(row['confidence']),
                'timestamp': float(row['timestamp'])
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    def export_csv(self, traffic_data, filename='forensic_export.csv'):
        """Export data in CSV format"""
        export_df = traffic_data[[
            'session_id', 'entry_node', 'exit_node', 
            'correlation_score', 'confidence', 'timestamp'
        ]].copy()
        
        export_df.to_csv(filename, index=False)
        return filename

# Add to app.py sidebar export section:
