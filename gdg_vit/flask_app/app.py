import sys
import os
import json
import time
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify

# --- Path Configuration ---
# Add the parent directory (gdg_vit) to sys.path to allow importing 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# --- Application Imports ---
try:
    from src.utilities import (
        generate_tsp_instance,
        solve_tsp_brute_force,
        GUROBI_AVAILABLE
    )
    from src.QAOA_qiskit import qaoa_tsp, QISKIT_AVAILABLE
except ImportError as e:
    print(f"Error importing src modules: {e}")

app = Flask(__name__)

# --- Helper Functions ---

def create_plot_tsp(coords, path, title):
    """
    Create a 2D plot of the TSP route.
    coords: Nx2 array of [x, y]
    path: List of visited node indices
    """
    # Create nodes scatter
    x_nodes = [c[0] for c in coords]
    y_nodes = [c[1] for c in coords]
    
    trace_nodes = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=14, color='#FF4B4B', line=dict(color='white', width=2)),
        text=[str(i) for i in range(len(coords))],
        textposition="top center",
        name='Cities'
    )
    
    # Create edges based on path
    edge_x = []
    edge_y = []
    
    # Path is a list of indices, e.g., [0, 2, 1, 3]
    # We need to close the loop
    full_path = path + [path[0]]
    
    for i in range(len(full_path) - 1):
        u, v = full_path[i], full_path[i+1]
        edge_x.append(coords[u][0])
        edge_x.append(coords[v][0])
        edge_x.append(None)
        edge_y.append(coords[u][1])
        edge_y.append(coords[v][1])
        edge_y.append(None)
        
    trace_edges = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='#0068C9', width=3),
        name='Route'
    )

    layout = go.Layout(
        title=title,
        showlegend=True,
        margin=dict(t=40, b=20, l=20, r=20),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )

    return go.Figure(data=[trace_edges, trace_nodes], layout=layout)

def create_plot_landscape_placeholder():
    # Placeholder since landscape is less relevant/harder to compute specific to TSP instance instantly
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2))
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.9)])
    fig.update_layout(
        title='Optimization Landscape (Conceptual)',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=300, margin=dict(t=30, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    
    # Extract params
    n_nodes = int(data.get('n_nodes', 4))
    # TSP scales as N^2 qubits. 
    # N=4 -> 16 qubits (OK for simulator/Brisbane)
    # N=5 -> 25 qubits (Pushing it for simulator, OK for Brisbane)
    # Clamp N to 5 to avoid crashing user's local sim or timing out
    if n_nodes > 5: n_nodes = 5 # Force limit for safety
    
    seed = int(data.get('seed', 42))
    quantum_backend = data.get('quantum_backend', 'simulator (local)')
    # Use hardcoded token in QAOA_qiskit if not provided here
    ibm_token = data.get('ibm_token', '') 
    
    # 1. Generate TSP Instance
    try:
        coords, dist_matrix = generate_tsp_instance(n_nodes, seed)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 2. Classical Algorithm (Brute Force for small N)
    start_c = time.time()
    c_cost, c_path = solve_tsp_brute_force(dist_matrix)
    end_c = time.time()
    
    # 3. Quantum Algorithm
    start_q = time.time()
    try:
        q_cost, q_path = qaoa_tsp(
            dist_matrix, 
            layer_count=1,
            backend_name=quantum_backend,
            ibm_token=ibm_token
        )
    except Exception as e:
         return jsonify({'error': f"Quantum simulation failed: {str(e)}"}), 500
    end_q = time.time()
    
    # 4. Prepare Plots
    
    # Classical Plot
    fig_classical = create_plot_tsp(coords, c_path, f"Classical Route (Cost: {c_cost:.2f})")
    
    # Quantum Plot
    fig_quantum = create_plot_tsp(coords, q_path, f"Quantum Route (Cost: {q_cost:.2f})")
    
    # Performance Chart
    fig_perf = go.Figure(data=[
        go.Bar(name='Classical', x=['Path Distance'], y=[c_cost], marker_color='#0068C9'),
        go.Bar(name='Quantum', x=['Path Distance'], y=[q_cost], marker_color='#FF4B4B')
    ])
    fig_perf.update_layout(
         barmode='group', title="Performance Comparison (Lower is Better)",
         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
         font=dict(color='white')
    )

    # Serialize Plots
    plots = {
        'classical': json.loads(json.dumps(fig_classical, cls=plotly.utils.PlotlyJSONEncoder)),
        'quantum': json.loads(json.dumps(fig_quantum, cls=plotly.utils.PlotlyJSONEncoder)),
        'landscape': json.loads(json.dumps(create_plot_landscape_placeholder(), cls=plotly.utils.PlotlyJSONEncoder)),
        'performance': json.loads(json.dumps(fig_perf, cls=plotly.utils.PlotlyJSONEncoder))
    }

    results = {
        'classical_cost': float(f"{c_cost:.2f}"),
        'classical_time': end_c - start_c,
        'quantum_cost': float(f"{q_cost:.2f}"),
        'quantum_time': end_q - start_q,
        'bitstring': str(q_path), # Reuse bitstring field for path string
        'actual_n': n_nodes,
        'plots': plots
    }
    
    # --- Firebase Integration ---
    try:
        from src.firebase_handler import save_experiment_result
        save_experiment_result(results)
    except Exception as e:
        print(f"Firebase save wrapper failed: {e}")

    return jsonify(results)

@app.route('/history', methods=['GET'])
def get_history():
    try:
        from src.firebase_handler import init_firebase
        db = init_firebase()
        if not db:
            return jsonify([])
        
        # Get last 10 experiments
        docs = db.collection('experiments').order_by('timestamp', direction='DESCENDING').limit(10).stream()
        history = []
        for doc in docs:
            d = doc.to_dict()
            # Remove plots to save bandwidth
            if 'plots' in d: del d['plots']
            history.append(d)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

