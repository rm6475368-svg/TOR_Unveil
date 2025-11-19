import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile

# Import your modules
from tor_collector import TORDataCollector
from correlation_engine import CorrelationEngine
from visualizer import NetworkVisualizer

# Main page config
st.set_page_config(
    page_title="TOR Unveil - Traffic Correlation System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'collector' not in st.session_state:
    st.session_state.collector = TORDataCollector()
    st.session_state.engine = CorrelationEngine()
    st.session_state.visualizer = NetworkVisualizer()
    st.session_state.traffic_data = None
    st.session_state.node_data = None

st.markdown('<h1 style="color:#1f77b4;text-align:center;">🔍 TOR UNVEIL - Peel the Onion</h1>', unsafe_allow_html=True)
st.markdown("### Analytical System to Trace TOR Network Users")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.title("Control Panel")
    st.subheader("1. Data Collection")
    if st.button("🔄 Fetch TOR Nodes"):
        with st.spinner("Fetching TOR node data..."):
            st.session_state.node_data = st.session_state.collector.fetch_exit_nodes()
            st.success(f"✅ Loaded {len(st.session_state.node_data)} nodes")
    if st.button("📊 Generate Traffic Patterns"):
        with st.spinner("Generating traffic correlation data..."):
            st.session_state.traffic_data = st.session_state.collector.generate_traffic_patterns()
            st.success(f"✅ Generated {len(st.session_state.traffic_data)} sessions")
    st.markdown("---")
    st.subheader("2. Analysis Settings")
    correlation_threshold = st.slider(
        "Correlation Threshold",
        min_value=0.5, max_value=1.0, value=0.75, step=0.05
    )
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.6, max_value=1.0, value=0.80, step=0.05
    )
    st.markdown("---")
    st.subheader("3. Export")
    if st.button("📥 Generate Forensic Report"):
        if st.session_state.traffic_data is not None:
            from report_generator import ForensicReportGenerator
            reporter = ForensicReportGenerator()
            report_md = reporter.generate_markdown_report(
                st.session_state.traffic_data,
                st.session_state.node_data,
                confidence_threshold
            )
            st.download_button(
                label="📄 Download Markdown Report",
                data=report_md,
                file_name=f"tor_forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
            st.success("✅ Report generated successfully!")
        else:
            st.warning("⚠️ Generate traffic data first!")

    # ---- Live PCAP/Network-Log Integration ----
    st.markdown("---")
    st.subheader("4. PCAP Upload (Real-World Data)")
    uploaded_file = st.file_uploader("Upload a PCAP file for dashboard analysis", type=["pcap", "pcapng"], key="main_pcap")
    if uploaded_file is not None:
        with st.spinner("Parsing uploaded PCAP..."):
            from pcap_collector import parse_pcap_to_sessions
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(uploaded_file.read())
                temp_pcap_path = tf.name
            st.session_state.traffic_data = parse_pcap_to_sessions(temp_pcap_path)
            st.success("PCAP parsed. Dashboard now shows real packet data.")

    # ---- Entry/Guard Node True Identification ----
    st.markdown("---")
    st.subheader("5. Entry/Guard Node True Identification")
    edge_file = st.file_uploader("Upload Edge PCAP (User/Network)", type=["pcap", "pcapng"], key="edge_pcap")
    guard_file = st.file_uploader("Upload Guard Node PCAP", type=["pcap", "pcapng"], key="guard_pcap")
    if edge_file is not None and guard_file is not None:
        from pcap_collector import parse_pcap_to_sessions
        from correlation_engine import correlate_entry_and_guard
        with tempfile.NamedTemporaryFile(delete=False) as tf1:
            tf1.write(edge_file.read())
            edge_pcap = tf1.name
        with tempfile.NamedTemporaryFile(delete=False) as tf2:
            tf2.write(guard_file.read())
            guard_pcap = tf2.name
        edge_df = parse_pcap_to_sessions(edge_pcap)
        guard_df = parse_pcap_to_sessions(guard_pcap)
        st.session_state.guard_results = correlate_entry_and_guard(edge_df, guard_df)
        st.success("Edge/Guard matching complete! See analysis below.")

# Dashboard tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview Dashboard",
    "🌐 Network Topology",
    "🎯 Correlation Analysis",
    "📈 Timeline Reconstruction",
    "🤖 ML Performance",
    "📋 Statistics & Rigor"
])

with tab1:
    st.subheader("System Overview & Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="TOR Nodes Monitored",
            value=len(st.session_state.node_data) if st.session_state.node_data is not None else 0,
            delta="Live"
        )
    with col2:
        st.metric(
            label="Active Sessions",
            value=len(st.session_state.traffic_data) if st.session_state.traffic_data is not None else 0,
            delta="+5"
        )
    with col3:
        if st.session_state.traffic_data is not None:
            high_conf = len(st.session_state.traffic_data[
                st.session_state.traffic_data['confidence'] >= confidence_threshold
            ])
            st.metric(
                label="High Confidence Matches",
                value=high_conf,
                delta=f"{(high_conf/len(st.session_state.traffic_data)*100):.1f}%"
            )
        else:
            st.metric(label="High Confidence Matches", value=0)
    with col4:
        st.metric(
            label="Avg Correlation Score",
            value=f"{st.session_state.traffic_data['correlation_score'].mean():.3f}" if st.session_state.traffic_data is not None else "0.000"
        )
    st.markdown("---")
    if st.session_state.traffic_data is not None:
        st.subheader("🔍 Detected Traffic Correlations")
        filtered_data = st.session_state.traffic_data[
            st.session_state.traffic_data['confidence'] >= confidence_threshold
        ].sort_values('confidence', ascending=False)
        display_cols = ['session_id', 'entry_node', 'exit_node', 'correlation_score', 'confidence']
        st.dataframe(
            filtered_data[display_cols],
            height=300
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                st.session_state.traffic_data,
                x='correlation_score',
                nbins=20,
                title='Correlation Score Distribution',
                labels={'correlation_score': 'Correlation Score'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(
                st.session_state.traffic_data,
                y='confidence',
                title='Confidence Level Distribution',
                color_discrete_sequence=['#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Click 'Generate Traffic Patterns' or upload a PCAP in the sidebar to load data")

with tab2:
    st.subheader("🌐 TOR Network Topology Visualization")
    if st.session_state.traffic_data is not None:
        fig = st.session_state.visualizer.generate_plotly_network(
            st.session_state.traffic_data
        )
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            unique_entry = st.session_state.traffic_data['entry_node'].nunique()
            st.metric("Unique Entry Nodes", unique_entry)
        with col2:
            unique_exit = st.session_state.traffic_data['exit_node'].nunique()
            st.metric("Unique Exit Nodes", unique_exit)
        with col3:
            total_connections = len(st.session_state.traffic_data)
            st.metric("Total Connections", total_connections)
        if st.session_state.node_data is not None:
            st.subheader("📍 Geographic Distribution of Nodes")
            country_dist = st.session_state.node_data['country'].value_counts()
            fig = px.bar(
                x=country_dist.index,
                y=country_dist.values,
                labels={'x': 'Country', 'y': 'Node Count'},
                title='TOR Nodes by Country',
                color=country_dist.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Generate traffic patterns or upload a PCAP first to visualize network topology")

with tab3:
    st.subheader("🎯 Traffic Correlation Analysis")

    # === ENTRY/GUARD MATCHING DISPLAY (new feature) ===
    st.markdown("#### Entry/Guard Node True Identification Results (PCAP-to-PCAP Matching)")
    if "guard_results" in st.session_state:
        st.dataframe(st.session_state.guard_results)
        st.markdown("---")

    # ---- legacy/session-based analysis below, as before ----
    if st.session_state.traffic_data is not None:
        session_options = st.session_state.traffic_data['session_id'].tolist()
        selected_session = st.selectbox("Select Session to Analyze", session_options)
        if selected_session:
            session_data = st.session_state.traffic_data[
                st.session_state.traffic_data['session_id'] == selected_session
            ].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 Entry Node Information")
                st.code(f"IP Address: {session_data['entry_node']}")
                st.code(f"Packet Count: {len(session_data['entry_pattern'])}")
            with col2:
                st.markdown("#### 🔴 Exit Node Information")
                st.code(f"IP Address: {session_data['exit_node']}")
                st.code(f"Packet Count: {len(session_data['exit_pattern'])}")
                st.code(f"Correlation Score: {session_data['correlation_score']:.3f}")
            st.markdown("### 📊 Identification Confidence")
            confidence_pct = session_data['confidence'] * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 📈 Traffic Pattern Comparison")
            fig = st.session_state.visualizer.create_timeline_reconstruction(session_data)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Generate traffic patterns or upload a PCAP to enable analysis")

# -- tabs 4-6 remain as your original, or follow your demo version (not reprinted for space) --

st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
    TOR Unveil System | TN Police Hackathon 2025<br>
    Automated TOR topology mapping and node correlation for forensic investigation
</div>
""", unsafe_allow_html=True)
