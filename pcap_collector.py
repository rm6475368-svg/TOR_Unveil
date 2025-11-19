# pcap_collector.py

from scapy.all import rdpcap, IP, TCP
import pandas as pd

def parse_pcap_to_sessions(pcap_path):
    """
    Parse a PCAP file into one session row per unique TCP flow.

    Each session contains:
    - session_id: unique identifier
    - entry_node: source IP address
    - exit_node: destination IP address
    - entry_pattern: list of packet timestamps (entry direction)
    - exit_pattern: same as entry_pattern (for demo -- real parsing may differ)
    - correlation_score: placeholder for your analytics
    - confidence: placeholder for your analytics
    """
    packets = rdpcap(pcap_path)
    flows = {}
    for pkt in packets:
        # Only include TCP/IP packets
        if IP in pkt and TCP in pkt:
            # Four-tuple identifies a flow (src, dst, sport, dport)
            key = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport)
            if key not in flows:
                flows[key] = []
            flows[key].append(pkt.time)
    sess_list = []
    for i, ((src, dst, sport, dport), times) in enumerate(flows.items()):
        sess_list.append({
            "session_id": f"pcap_{i}",
            "entry_node": src,
            "exit_node": dst,
            "entry_pattern": times,      # List of timestamps
            "exit_pattern": times,       # For demo, use the same (or adapt for other direction)
            "correlation_score": 1.0,    # Placeholder: fill with real analysis later
            "confidence": 1.0            # Placeholder: fill with real analysis later
        })
    return pd.DataFrame(sess_list)

# Optional: if you want a CLI/demo utility
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pcap_collector.py <file.pcap>")
    else:
        df = parse_pcap_to_sessions(sys.argv[1])
        print(df)
