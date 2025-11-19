import networkx as nx
import plotly.graph_objects as go
import pandas as pd

class NetworkVisualizer:
    def __init__(self):
        self.G = nx.DiGraph()
    
    def create_network_graph(self, traffic_data):
        """Create network topology visualization"""
        self.G.clear()
        
        # Add nodes and edges from traffic data
        for _, session in traffic_data.iterrows():
            self.G.add_edge(
                session['entry_node'], 
                session['exit_node'],
                weight=session['correlation_score']
            )
        
        return self.G
    
    def generate_plotly_network(self, traffic_data):
        """Generate interactive Plotly network graph"""
        G = self.create_network_graph(traffic_data)
        
        # Generate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            # FIXED: Access pos with individual nodes, not the edge tuple
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]]['weight'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Node: {node}<br>Connections: {G.degree(node)}")
            node_color.append(G.degree(node))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=20,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left'
                ),
                line=dict(width=2, color='white')
            ))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='TOR Network Topology & Correlation Map',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        return fig
    
    def create_timeline_reconstruction(self, session_data):
        """Create timeline visualization of traffic flow"""
        fig = go.Figure()
        
        entry_pattern = session_data['entry_pattern']
        exit_pattern = session_data['exit_pattern']
        
        time_points = list(range(len(entry_pattern)))
        
        fig.add_trace(go.Scatter(
            x=time_points, 
            y=entry_pattern,
            mode='lines',
            name='Entry Node Traffic',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points, 
            y=exit_pattern,
            mode='lines',
            name='Exit Node Traffic',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Traffic Pattern Timeline - Entry vs Exit',
            xaxis_title='Time (ms)',
            yaxis_title='Packet Delay',
            hovermode='x unified',
            height=400
        )
        
        return fig
