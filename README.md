TOR Unveil – Traffic Correlation System

A prototype for entry/exit node correlation and user de-anonymization on the TOR network
Built for TN Police Hackathon 2025
📌 Overview

TOR Unveil is a forensic analytics dashboard that demonstrates how investigators might use traffic pattern correlation to link the origin and destination of users on the TOR network.
This prototype simulates the workflow of real-world traffic analysis, providing visualizations, machine learning confidence scores, and downloadable forensic reports—all through an interactive web dashboard.
🚀 Features

    Collect live TOR node lists

    Simulate or analyze TOR entry/exit traffic patterns

    Statistical and ML-based traffic correlation (Random Forest)

    Interactive dashboard: node graphs, timelines, session drill-downs

    Forensic report generation (Markdown export)

    Six analytical tabs: Overview, Topology, Correlation, Timeline, ML, Statistics

🛠️ Installation

    Clone the repository

bash
git clone 
cd tor-unveil-hackathon

Install requirements

bash
pip install streamlit pandas plotly networkx requests scikit-learn scipy

Run the dashboard

    bash
    streamlit run app.py

    Open your browser to http://localhost:8501

🔎 Usage

    Fetch TOR Nodes – Use the sidebar in the dashboard to get fresh node lists.

    Generate Traffic Patterns – Load simulated traffic flows for analysis.

    Explore the Tabs:

        Overview: Metrics, session matches, correlation scores

        Network Topology: Interactive entry→exit graph

        Correlation Analysis: Drill into individual traffic sessions

        Timeline Reconstruction: Visualize flow over time

        ML Performance: Model accuracy, feature importance

        Statistics & Rigor: Confidence intervals, p-values, effect size

    Export Reports – Generate and download a Markdown forensic summary via sidebar.

📂 File Structure

text
tor-unveil-hackathon/
├── app.py
├── tor_collector.py
├── correlation_engine.py
├── visualizer.py
├── report_generator.py
├── data/                  # (Optional: sample/test data)
└── README.md

⚡ Architecture

    Data collection (TOR node API)

    Traffic simulation or PCAP flow (conceptual, can be replaced by real PCAP)

    Statistical & ML-based entry/exit correlation

    NetworkX + Plotly streamlit visualizations

    Forensic reporting (Markdown export)

⚠️ Limitations

    Traffic data is simulated for this demonstration. Real PCAP integration is an easy extension.

    User de-anonymization is not run against the actual TOR network—the project demonstrates how it would work in law enforcement with live data.

    ML model uses static (not live-trained) data for hackathon speed.

| Requirement               | Implemented |
| ------------------------- | ----------- |
| TOR Data Collection       | ✅           |
| Node Correlation Engine   | ✅           |
| Entry Node Identification | ✅ (Demo)    |
| Visualizations            | ✅           |
| Forensic Reporting        | ✅           |
| Machine Learning/Stats    | ✅           |



🧭 Future Work

    Direct real-time PCAP/Netflow analysis

    Deep learning architecture (e.g., DeepCorr/CoFFEA models)

    Distributed, scalable deployment

    Integration with legal compliance/audit frameworks

👥 Credits

Built by [Rajasekar V/ TOR Unveil]
TN Police Hackathon 2025

For demo, academic, and training use only.
