from typing import Dict, List, Any
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from dash import Dash, html, dcc, Input, Output, dash_table


class VisualizationDashboard:
    """
    Class for creating visualizations and dashboards of chunking strategy evaluations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dashboard with configuration.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.output_dir = config.get("output_directory", "./results/")
        self.formats = config.get("save_format", ["png", "html"])
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Dashboard settings
        self.dashboard_config = config.get("dashboard", {})
        self.port = self.dashboard_config.get("port", 8050)
        self.debug = self.dashboard_config.get("debug", False)
        self.show_chunks = self.dashboard_config.get("show_individual_chunks", True)
        
        # Store for results
        self.results = None
        self.aggregated = None
        self.best_strategy = None
    
    def save_results(self, evaluation_results: Dict[str, Dict[str, Dict[str, Any]]], 
                    aggregated_metrics: Dict[str, Dict[str, Any]], 
                    best_strategy: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            evaluation_results: Results from evaluator
            aggregated_metrics: Aggregated metrics
            best_strategy: Name of the best strategy
        """
        self.results = evaluation_results
        self.aggregated = aggregated_metrics
        self.best_strategy = best_strategy
        
        # Remove any embedded arrays/vectors from results (for clean JSON)
        clean_results = self._clean_for_json(evaluation_results)
        clean_aggregated = self._clean_for_json(aggregated_metrics)
        
        # Create a combined results file
        combined = {
            "evaluation_results": clean_results,
            "aggregated_metrics": clean_aggregated,
            "best_strategy": best_strategy
        }
        
        # Save to file
        with open(os.path.join(self.output_dir, "evaluation_results.json"), 'w') as f:
            json.dump(combined, f, indent=2)
    
    def generate_matplotlib_charts(self):
        """
        Generate and save visualization charts using Matplotlib.
        """
        if not self.aggregated:
            print("No results to visualize.")
            return
        
        # 1. Generate context precision chart
        self._generate_precision_chart()
        
        # 2. Generate token efficiency chart
        self._generate_token_efficiency_chart()
        
        # 3. Generate time comparison chart
        self._generate_time_chart()
        
        # 4. Generate overall performance chart
        self._generate_overall_chart()
        
        # 5. Generate chunk distribution chart
        if self.results:
            self._generate_chunk_distribution_chart()
    
    def generate_plotly_charts(self):
        """
        Generate and save interactive visualization charts using Plotly.
        """
        if not self.aggregated:
            print("No results to visualize.")
            return
        
        # 1. Generate precision and recall comparison
        self._generate_plotly_precision_chart()
        
        # 2. Generate token efficiency comparison
        self._generate_plotly_token_efficiency_chart()
        
        # 3. Generate time comparison
        self._generate_plotly_time_chart()
        
        # 4. Generate radar chart for overall comparison
        self._generate_plotly_radar_chart()
        
        # 5. Generate combined metrics chart
        self._generate_plotly_combined_metrics()
    
    def run_dashboard(self, evaluation_results=None, aggregated_metrics=None, best_strategy=None):
        """
        Run an interactive Dash dashboard with the evaluation results.
        
        Args:
            evaluation_results: Optional evaluation results to display
            aggregated_metrics: Optional aggregated metrics to display
            best_strategy: Optional name of the best strategy
        """
        # Use provided results or fall back to stored results
        results = evaluation_results if evaluation_results is not None else self.results
        aggregated = aggregated_metrics if aggregated_metrics is not None else self.aggregated
        best = best_strategy if best_strategy is not None else self.best_strategy
        
        # Update stored results if provided
        if evaluation_results is not None:
            self.results = evaluation_results
        if aggregated_metrics is not None:
            self.aggregated = aggregated_metrics
        if best_strategy is not None:
            self.best_strategy = best_strategy
        
        if not results or not aggregated:
            print("No results to display in dashboard.")
            return
        
        # Create app
        app = Dash(__name__, 
                 external_stylesheets=[
                     'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap',
                 ])
        
        # Convert aggregated metrics to DataFrame for easier handling
        strategies = list(aggregated.keys())
        
        # Make sure we have all the expected keys in the aggregated metrics
        normalized_metrics = {}
        for strategy, metrics in aggregated.items():
            normalized_metrics[strategy] = {
                "context_precision": metrics.get("context_precision", metrics.get("avg_context_precision", 0)),
                "token_efficiency": metrics.get("token_efficiency", metrics.get("avg_token_efficiency", 0)),
                "answer_relevance": metrics.get("answer_relevance", metrics.get("avg_answer_relevance", 0)),
                "latency": metrics.get("latency", metrics.get("avg_total_time", metrics.get("avg_latency", 0))),
                "combined_score": metrics.get("combined_score", metrics.get("avg_combined_score", 0)),
                "chunk_similarities": metrics.get("chunk_similarities", metrics.get("avg_chunk_similarities", 0))
            }
        
        metrics_df = pd.DataFrame(normalized_metrics).T.reset_index()
        metrics_df = metrics_df.rename(columns={"index": "strategy"})
        
        # Create dropdown options for queries
        queries = list(results.keys())
        query_options = [{"label": results[q]["query_text"][:50] + "..." if len(results[q]["query_text"]) > 50 else results[q]["query_text"], 
                         "value": q} for q in queries]
        
        # Define custom CSS for the dashboard
        app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>RAG Chunking Evaluation Dashboard</title>
                {%favicon%}
                {%css%}
                <style>
                    :root {
                        --primary-color: #4361ee;
                        --secondary-color: #3a0ca3;
                        --success-color: #4cc9f0;
                        --warning-color: #f72585;
                        --light-bg: #f8f9fa;
                        --dark-text: #212529;
                        --light-text: #f8f9fa;
                        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    
                    body {
                        font-family: 'Inter', sans-serif;
                        background-color: #f5f7fa;
                        color: var(--dark-text);
                        margin: 0;
                        padding: 0;
                    }
                    
                    .container {
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    
                    .header {
                        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
                        color: white;
                        padding: 20px 30px;
                        border-radius: 12px;
                        margin-bottom: 30px;
                        box-shadow: var(--card-shadow);
                    }
                    
                    .header h1 {
                        margin: 0;
                        font-weight: 700;
                        font-size: 28px;
                    }
                    
                    .card {
                        background: white;
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 25px;
                        box-shadow: var(--card-shadow);
                    }
                    
                    .card-title {
                        font-size: 18px;
                        font-weight: 600;
                        margin-top: 0;
                        margin-bottom: 20px;
                        color: var(--primary-color);
                        display: flex;
                        align-items: center;
                    }
                    
                    .card-title i {
                        margin-right: 8px;
                    }
                    
                    .status-card {
                        background: white;
                        border-radius: 12px;
                        padding: 15px;
                        display: flex;
                        flex-direction: column;
                        box-shadow: var(--card-shadow);
                    }
                    
                    .status-card .label {
                        font-size: 14px;
                        color: #6c757d;
                        margin-bottom: 5px;
                    }
                    
                    .status-card .value {
                        font-size: 22px;
                        font-weight: 600;
                        color: var(--primary-color);
                    }
                    
                    .best-strategy-card {
                        background: linear-gradient(145deg, #a8ff78, #78ffd6);
                        color: #2b2b2b;
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 25px;
                        box-shadow: var(--card-shadow);
                    }
                    
                    .best-strategy-card h2 {
                        margin-top: 0;
                        margin-bottom: 10px;
                        font-size: 20px;
                        font-weight: 600;
                    }
                    
                    .best-strategy-card h3 {
                        font-size: 28px;
                        margin: 10px 0;
                    }
                    
                    .best-strategy-card p {
                        margin: 0;
                        font-size: 15px;
                    }
                    
                    .section-title {
                        font-size: 22px;
                        font-weight: 600;
                        margin: 30px 0 20px 0;
                        color: var(--secondary-color);
                    }
                    
                    .tabs-container .tab-content {
                        background: white;
                        padding: 20px;
                        border-radius: 0 0 12px 12px;
                        border-top: none;
                    }
                    
                    .dash-table-container {
                        overflow-x: auto;
                    }
                    
                    .dash-table {
                        border-collapse: collapse;
                        width: 100%;
                    }
                    
                    .dash-table th {
                        background-color: #f8f9fa;
                        padding: 12px 15px;
                        text-align: left;
                        font-weight: 600;
                    }
                    
                    .dash-table td {
                        padding: 10px 15px;
                        border-top: 1px solid #e9ecef;
                    }
                    
                    .query-dropdown-container {
                        background: white;
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: var(--card-shadow);
                    }
                    
                    .dropdown-label {
                        font-weight: 500;
                        margin-bottom: 10px;
                        display: block;
                    }
                    
                    .Select-control {
                        border-radius: 8px;
                        border: 1px solid #ced4da;
                    }
                    
                    .Select-control:hover {
                        border-color: var(--primary-color);
                    }
                    
                    /* Dashboard tabs styling */
                    .dash-tab {
                        padding: 15px 20px;
                        font-weight: 500;
                    }
                    
                    .dash-tab--selected {
                        background-color: white;
                        color: var(--primary-color);
                        border-left: 1px solid #dee2e6;
                        border-right: 1px solid #dee2e6;
                        border-top: 3px solid var(--primary-color);
                        border-bottom: none;
                    }
                    
                    /* Highlight row for best strategy */
                    .highlight-row {
                        background-color: rgba(168, 255, 120, 0.2);
                        font-weight: 500;
                    }
                    
                    /* Strategy metrics cards */
                    .metrics-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                        gap: 20px;
                        margin-bottom: 25px;
                    }
                    
                    /* Responsive adjustments */
                    @media (max-width: 768px) {
                        .metrics-grid {
                            grid-template-columns: 1fr;
                        }
                        
                        .header {
                            padding: 15px;
                        }
                        
                        .header h1 {
                            font-size: 22px;
                        }
                    }
                </style>
                {%scripts%}
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Layout
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("RAG Chunking Strategy Evaluation Dashboard")
            ], className="header"),
            
            # Main content container
            html.Div([
                # Top metrics summary
                html.Div([
                    # Best strategy card
                    html.Div([
                        html.H2("Best Performing Strategy"),
                        html.H3(best),
                        html.P(f"Based on comprehensive evaluation across multiple queries and metrics")
                    ], className="best-strategy-card"),
                    
                    # Key metrics grid
                    html.Div([
                        # Count of strategies
                        html.Div([
                            html.Div("Strategies Evaluated", className="label"),
                            html.Div(f"{len(strategies)}", className="value")
                        ], className="status-card"),
                        
                        # Count of queries
                        html.Div([
                            html.Div("Queries Analyzed", className="label"),
                            html.Div(f"{len(queries)}", className="value")
                        ], className="status-card"),
                        
                        # Best score
                        html.Div([
                            html.Div("Top Combined Score", className="label"),
                            html.Div(f"{normalized_metrics[best]['combined_score']:.2f}" if best in normalized_metrics else "N/A", className="value")
                        ], className="status-card"),
                    ], className="metrics-grid")
                ]),
                
                # Aggregated Performance section
                html.H2("Aggregated Performance Metrics", className="section-title"),
                
                # Tabs for different metric views
                html.Div([
                    dcc.Tabs([
                        dcc.Tab(
                            label="📊 Metrics Table", 
                            children=[
                                dash_table.DataTable(
                                    id="metrics-table",
                                    columns=[
                                        {"name": "Strategy", "id": "strategy"},
                                        {"name": "Context Precision", "id": "context_precision", "format": {"specifier": ".4f"}},
                                        {"name": "Token Efficiency", "id": "token_efficiency", "format": {"specifier": ".4f"}},
                                        {"name": "Answer Relevance", "id": "answer_relevance", "format": {"specifier": ".4f"}},
                                        {"name": "Latency (s)", "id": "latency", "format": {"specifier": ".2f"}},
                                        {"name": "Combined Score", "id": "combined_score", "format": {"specifier": ".4f"}},
                                    ],
                                    data=metrics_df.to_dict("records"),
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": metrics_df[metrics_df["strategy"] == best].index[0] if best in metrics_df["strategy"].values else -1},
                                            "backgroundColor": "rgba(168, 255, 120, 0.2)",
                                            "fontWeight": "500"
                                        }
                                    ],
                                    style_header={
                                        'backgroundColor': '#f8f9fa',
                                        'fontWeight': 'bold',
                                        'border': '1px solid #ddd'
                                    },
                                    style_cell={
                                        'padding': '12px 15px',
                                        'textAlign': 'left',
                                        'fontFamily': 'Inter, sans-serif'
                                    },
                                    style_data={
                                        'border': '1px solid #f0f0f0'
                                    },
                                    sort_action="native",
                                    sort_mode="multi",
                                    style_table={"overflowX": "auto"}
                                )
                            ],
                            className="dash-tab",
                            selected_className="dash-tab--selected"
                        ),
                        
                        dcc.Tab(
                            label="📈 Precision & Efficiency", 
                            children=[
                                dcc.Graph(
                                    id="precision-efficiency-chart",
                                    figure=self._generate_plotly_precision_chart(return_fig=True)
                                )
                            ],
                            className="dash-tab",
                            selected_className="dash-tab--selected"
                        ),
                        
                        dcc.Tab(
                            label="⏱️ Time & Tokens", 
                            children=[
                                dcc.Graph(
                                    id="time-tokens-chart",
                                    figure=self._generate_plotly_time_chart(return_fig=True)
                                )
                            ],
                            className="dash-tab",
                            selected_className="dash-tab--selected"
                        ),
                        
                        dcc.Tab(
                            label="🎯 Overall Comparison", 
                            children=[
                                dcc.Graph(
                                    id="radar-chart",
                                    figure=self._generate_plotly_radar_chart(return_fig=True)
                                )
                            ],
                            className="dash-tab",
                            selected_className="dash-tab--selected"
                        ),
                    ], className="tabs-container")
                ], className="card"),
                
                # Query-specific results section
                html.H2("Query-Specific Results", className="section-title"),
                
                # Query selection dropdown
                html.Div([
                    html.Label("Select a query to analyze:", className="dropdown-label"),
                    dcc.Dropdown(
                        id="query-dropdown",
                        options=query_options,
                        value=queries[0] if queries else None,
                        clearable=False
                    )
                ], className="query-dropdown-container"),
                
                # Query results visualization
                html.Div([
                    dcc.Graph(id="query-metrics-chart")
                ], className="card"),
                
                # Retrieved chunks section
                html.Div([
                    html.H2("Retrieved Chunks", className="card-title"),
                    html.Div(id="chunks-container")
                ], className="card") if self.show_chunks else html.Div(),
                
                # Generated answers section
                html.Div([
                    html.H2("Generated Answers", className="card-title"),
                    html.Div(id="answers-container")
                ], className="card")
                
            ], className="container")
        ])
        
        # Callbacks
        @app.callback(
            [Output("query-metrics-chart", "figure"),
             Output("chunks-container", "children"),
             Output("answers-container", "children")],
            [Input("query-dropdown", "value")]
        )
        def update_query_results(query):
            if not query or query not in results:
                return {}, [], []
            
            # Generate chart for this query's metrics
            query_results = results[query]
            
            # Prepare data for the chart
            data = []
            for strategy, metrics in query_results["results"].items():
                if "error" not in metrics:
                    data.append({
                        "strategy": strategy,
                        "retrieval_time": metrics.get("latency", 0),
                        "context_precision": metrics.get("context_precision", 0),
                        "token_efficiency": metrics.get("token_efficiency", 0),
                        "answer_relevance": metrics.get("answer_relevance", 0),
                        "combined_score": metrics.get("combined_score", 0),
                        "context_tokens": metrics.get("context_tokens", 0)
                    })
            
            # Create chart
            if not data:
                return {}, [], []
            
            df = pd.DataFrame(data)
            
            # Color scheme
            colors = {
                best: '#4cc9f0',  # Best strategy color
                'default': ['#4361ee', '#3a0ca3', '#7209b7', '#f72585']  # Colors for other strategies
            }
            
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                subplot_titles=("Context Precision", "Token Efficiency", 
                               "Retrieval Time (s)", "Context Tokens"),
                vertical_spacing=0.16
            )
            
            # Create color map for strategies
            color_map = {}
            for i, strategy in enumerate(df["strategy"]):
                if strategy == best:
                    color_map[strategy] = colors[best]
                else:
                    color_map[strategy] = colors['default'][i % len(colors['default'])]
            
            # Add bars for each metric
            fig.add_trace(
                go.Bar(
                    x=df["strategy"], 
                    y=df["context_precision"], 
                    name="Context Precision",
                    marker_color=[color_map[s] for s in df["strategy"]],
                    hovertemplate='<b>%{x}</b><br>Precision: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df["strategy"], 
                    y=df["token_efficiency"], 
                    name="Token Efficiency",
                    marker_color=[color_map[s] for s in df["strategy"]],
                    hovertemplate='<b>%{x}</b><br>Efficiency: %{y:.4f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=df["strategy"], 
                    y=df["retrieval_time"], 
                    name="Retrieval Time",
                    marker_color=[color_map[s] for s in df["strategy"]],
                    hovertemplate='<b>%{x}</b><br>Time: %{y:.2f}s<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df["strategy"], 
                    y=df["context_tokens"], 
                    name="Context Tokens",
                    marker_color=[color_map[s] for s in df["strategy"]],
                    hovertemplate='<b>%{x}</b><br>Tokens: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                title={
                    'text': f"Performance Metrics for Query",
                    'y': 0.98,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=18, family='Inter, sans-serif')
                },
                showlegend=False,
                font=dict(family='Inter, sans-serif'),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=60, r=60, t=100, b=60)
            )
            
            # Add query as subtitle
            fig.add_annotation(
                text=f"<i>\"{query_results['query_text'][:75]}...\"</i>" if len(query_results['query_text']) > 75 else f"<i>\"{query_results['query_text']}\"</i>",
                xref="paper", yref="paper",
                x=0.5, y=0.99,
                showarrow=False,
                font=dict(size=12, color="#666666", family='Inter, sans-serif'),
                xanchor='center'
            )
            
            # Update axis properties
            fig.update_xaxes(
                tickangle=30,
                tickfont=dict(size=10),
                gridcolor='#f0f0f0'
            )
            
            fig.update_yaxes(
                gridcolor='#f0f0f0',
                tickfont=dict(size=10)
            )
            
            # Generate chunks display
            chunks_display = []
            for strategy, metrics in query_results["results"].items():
                if "error" not in metrics:
                    # Try to get chunks from different possible locations in the data structure
                    chunks = []
                    if "chunks" in metrics:
                        chunks = metrics["chunks"]
                    elif "similar_chunks" in metrics:
                        chunks = metrics["similar_chunks"]
                    elif "chunks_retrieved" in metrics and isinstance(metrics["chunks_retrieved"], list):
                        chunks = metrics["chunks_retrieved"]
                    
                    if chunks:
                        # Create table rows for each chunk
                        rows = []
                        for i, chunk in enumerate(chunks):
                            similarity = (f"{chunk.get('similarity', 0):.4f}" 
                                        if isinstance(chunk, dict) and 'similarity' in chunk 
                                        else "N/A")
                            
                            # Get text content with proper handling
                            if isinstance(chunk, dict):
                                text = chunk.get("text", "")
                            elif isinstance(chunk, str):
                                text = chunk[:100] + "..." if len(chunk) > 100 else chunk
                            else:
                                text = str(chunk)[:100]
                            
                            # Add row with modern styling
                            rows.append(
                                html.Tr([
                                    html.Td(i+1, style={"width": "80px", "textAlign": "center"}),
                                    html.Td(similarity, style={"width": "120px", "textAlign": "center"}),
                                    html.Td(text, style={"wordBreak": "break-word"})
                                ], style={"borderBottom": "1px solid #e0e0e0"})
                            )
                        
                        # Add strategy with chunks
                        chunks_display.append(
                            html.Div([
                                # Strategy header with badge
                                html.Div([
                                    html.Span(
                                        strategy, 
                                        style={
                                            "backgroundColor": color_map.get(strategy, "#4361ee"),
                                            "color": "white",
                                            "padding": "6px 12px",
                                            "borderRadius": "16px",
                                            "fontWeight": "500",
                                            "fontSize": "14px",
                                            "display": "inline-block"
                                        }
                                    ),
                                    html.Span(
                                        f"{len(chunks)} chunks retrieved", 
                                        style={
                                            "marginLeft": "10px", 
                                            "fontSize": "14px",
                                            "color": "#666"
                                        }
                                    )
                                ], style={"marginBottom": "15px"}),
                                
                                # Chunks table
                                html.Table([
                                    html.Thead(
                                        html.Tr([
                                            html.Th("Chunk #", style={"width": "80px", "textAlign": "center"}),
                                            html.Th("Similarity", style={"width": "120px", "textAlign": "center"}),
                                            html.Th("Text Preview")
                                        ], style={"backgroundColor": "#f8f9fa", "borderBottom": "2px solid #dee2e6"})
                                    ),
                                    html.Tbody(rows)
                                ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "14px"})
                            ], style={"marginBottom": "30px"})
                        )
            
            # If no chunks found, display a message
            if not chunks_display:
                chunks_display = [
                    html.Div(
                        html.P("No chunk data available for this query", 
                              style={"color": "#666", "fontStyle": "italic", "textAlign": "center"}),
                        style={"padding": "20px"}
                    )
                ]
            
            # Generate answers display
            answers_display = []
            for strategy, metrics in query_results["results"].items():
                if "error" not in metrics and "answer" in metrics:
                    answer = metrics["answer"]
                    
                    # Create answer display with modern styling
                    answers_display.append(
                        html.Div([
                            # Strategy header with badge
                            html.Div([
                                html.Span(
                                    strategy, 
                                    style={
                                        "backgroundColor": color_map.get(strategy, "#4361ee"),
                                        "color": "white",
                                        "padding": "6px 12px",
                                        "borderRadius": "16px",
                                        "fontWeight": "500",
                                        "fontSize": "14px",
                                        "display": "inline-block"
                                    }
                                )
                            ], style={"marginBottom": "15px"}),
                            
                            # Answer content
                            html.Div(
                                html.P(answer, style={"margin": "0", "lineHeight": "1.6"}),
                                style={
                                    "border": "1px solid #e0e0e0", 
                                    "padding": "15px", 
                                    "borderRadius": "8px",
                                    "backgroundColor": "#f9f9f9",
                                    "fontSize": "14px"
                                }
                            )
                        ], style={"marginBottom": "30px"})
                    )
            
            # If no answers found, display a message
            if not answers_display:
                answers_display = [
                    html.Div(
                        html.P("No answer data available for this query", 
                              style={"color": "#666", "fontStyle": "italic", "textAlign": "center"}),
                        style={"padding": "20px"}
                    )
                ]
            
            return fig, chunks_display, answers_display
        
        # Run server
        app.run_server(debug=self.debug, port=self.port)
    
    def generate_final_report(self):
        """
        Generate a combined final report with metrics and visualizations.
        """
        if not self.aggregated:
            print("No results to include in report.")
            return
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chunking Strategy Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .verdict {{ background-color: #e6ffe6; padding: 15px; border-radius: 5px; border: 1px solid #009900; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ margin-bottom: 20px; }}
                .best {{ font-weight: bold; background-color: #e6ffe6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Chunking Strategy Evaluation Report</h1>
                
                <div class="verdict">
                    <h2>Final Verdict</h2>
                    <p>Based on comprehensive evaluation, the <strong>{self.best_strategy}</strong> chunking strategy performed best overall.</p>
                </div>
                
                <h2>Aggregated Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Context Precision</th>
                        <th>Token Efficiency</th>
                        <th>Retrieval Time (s)</th>
                        <th>Generation Time (s)</th>
                        <th>Total Time (s)</th>
                        <th>Context Tokens</th>
                    </tr>
        """
        
        # Add rows for each strategy
        for strategy, metrics in self.aggregated.items():
            row_class = "best" if strategy == self.best_strategy else ""
            html_content += f"""
                    <tr class="{row_class}">
                        <td>{strategy}</td>
                        <td>{metrics.get("avg_context_precision", 0):.4f}</td>
                        <td>{metrics.get("avg_token_efficiency", 0):.4f}</td>
                        <td>{metrics.get("avg_retrieval_time", 0):.4f}</td>
                        <td>{metrics.get("avg_generation_time", 0):.4f}</td>
                        <td>{metrics.get("avg_total_time", 0):.4f}</td>
                        <td>{metrics.get("avg_context_tokens", 0):.0f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>Performance Visualizations</h2>
                <div class="chart">
                    <h3>Precision and Efficiency</h3>
                    <img src="precision_chart.png" alt="Precision and Efficiency Chart" style="max-width: 100%;">
                </div>
                
                <div class="chart">
                    <h3>Time and Token Usage</h3>
                    <img src="time_chart.png" alt="Time and Token Usage Chart" style="max-width: 100%;">
                </div>
                
                <div class="chart">
                    <h3>Overall Strategy Comparison</h3>
                    <img src="overall_chart.png" alt="Overall Strategy Comparison" style="max-width: 100%;">
                </div>
                
                <h2>Conclusion</h2>
                <p>
                    This evaluation compared different chunking strategies for a RAG pipeline to determine which performs best
                    across multiple metrics including retrieval precision, token efficiency, and processing time.
                    The <strong>{self.best_strategy}</strong> strategy demonstrated the best overall performance based on these metrics.
                </p>
                
                <p>
                    For a more detailed analysis, including per-query performance and interactive visualizations,
                    launch the dashboard using <code>python app/main.py</code>.
                </p>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        with open(os.path.join(self.output_dir, "evaluation_report.html"), 'w') as f:
            f.write(html_content)
    
    def _clean_for_json(self, data):
        """
        Clean data for JSON serialization by removing numpy arrays, etc.
        
        Args:
            data: Data to clean
            
        Returns:
            Cleaned data suitable for JSON serialization
        """
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.number):
            return float(data)
        else:
            return data
    
    def _generate_precision_chart(self):
        """Generate and save context precision chart."""
        plt.figure(figsize=(10, 6))
        
        strategies = []
        precision_values = []
        
        for strategy, metrics in self.aggregated.items():
            strategies.append(strategy)
            precision_values.append(metrics.get("avg_context_precision", 0))
        
        bars = plt.bar(strategies, precision_values, color='skyblue')
        
        # Highlight best strategy
        if self.best_strategy and self.best_strategy in strategies:
            best_idx = strategies.index(self.best_strategy)
            bars[best_idx].set_color('green')
        
        plt.title('Average Context Precision by Chunking Strategy')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('Precision (0-1)')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(self.output_dir, "precision_chart.png"))
        plt.close()
    
    def _generate_token_efficiency_chart(self):
        """Generate and save token efficiency chart."""
        plt.figure(figsize=(10, 6))
        
        strategies = []
        efficiency_values = []
        
        for strategy, metrics in self.aggregated.items():
            strategies.append(strategy)
            efficiency_values.append(metrics.get("avg_token_efficiency", 0))
        
        bars = plt.bar(strategies, efficiency_values, color='lightgreen')
        
        # Highlight best strategy
        if self.best_strategy and self.best_strategy in strategies:
            best_idx = strategies.index(self.best_strategy)
            bars[best_idx].set_color('green')
        
        plt.title('Average Token Efficiency by Chunking Strategy')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('Efficiency (0-1)')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(self.output_dir, "token_efficiency_chart.png"))
        plt.close()
    
    def _generate_time_chart(self):
        """Generate and save time comparison chart."""
        plt.figure(figsize=(12, 6))
        
        strategies = []
        retrieval_times = []
        generation_times = []
        
        for strategy, metrics in self.aggregated.items():
            strategies.append(strategy)
            retrieval_times.append(metrics.get("avg_retrieval_time", 0))
            generation_times.append(metrics.get("avg_generation_time", 0))
        
        width = 0.35
        x = np.arange(len(strategies))
        
        plt.bar(x - width/2, retrieval_times, width, label='Retrieval Time', color='coral')
        plt.bar(x + width/2, generation_times, width, label='Generation Time', color='lightblue')
        
        plt.title('Average Processing Times by Chunking Strategy')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('Time (seconds)')
        plt.xticks(x, strategies, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(self.output_dir, "time_chart.png"))
        plt.close()
    
    def _generate_overall_chart(self):
        """Generate and save overall performance chart."""
        plt.figure(figsize=(10, 8))
        
        # Get the metrics we want to visualize
        strategies = list(self.aggregated.keys())
        metrics = ["avg_context_precision", "avg_token_efficiency", "avg_chunk_similarities"]
        
        # Create data for the chart
        data = []
        for strategy in strategies:
            values = [self.aggregated[strategy].get(metric, 0) for metric in metrics]
            data.append(values)
        
        # Normalize time (lower is better)
        time_values = [self.aggregated[strategy].get("avg_total_time", 0) for strategy in strategies]
        if any(time_values):
            max_time = max(time_values)
            normalized_times = [1 - (t / max_time) if max_time > 0 else 0 for t in time_values]
            for i, values in enumerate(data):
                data[i].append(normalized_times[i])
            metrics.append("normalized_time")
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        for i, strategy in enumerate(strategies):
            values = data[i]
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, label=strategy)
            ax.fill(angles, values, alpha=0.1)
        
        # Set chart labels
        metric_labels = ["Context Precision", "Token Efficiency", "Chunk Similarities", "Time Efficiency"]
        plt.xticks(angles[:-1], metric_labels)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
        plt.ylim(0, 1)
        
        plt.title('Overall Performance Comparison', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "overall_chart.png"))
        plt.close()
    
    def _generate_chunk_distribution_chart(self):
        """Generate and save chunk distribution chart."""
        # This requires more detailed per-chunk data
        # For now, we'll use a simple approach based on what we have
        
        # Count chunks per strategy
        chunk_counts = {}
        for query, query_results in self.results.items():
            for strategy, result in query_results.items():
                if "error" not in result:
                    if strategy not in chunk_counts:
                        chunk_counts[strategy] = 0
                    chunk_counts[strategy] += result.get("retrieved_chunks", 0)
        
        if not chunk_counts:
            return
        
        plt.figure(figsize=(10, 6))
        
        strategies = list(chunk_counts.keys())
        counts = [chunk_counts[s] for s in strategies]
        
        bars = plt.bar(strategies, counts, color='lightblue')
        
        # Highlight best strategy
        if self.best_strategy and self.best_strategy in strategies:
            best_idx = strategies.index(self.best_strategy)
            bars[best_idx].set_color('green')
        
        plt.title('Total Retrieved Chunks by Strategy')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('Number of Chunks')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(os.path.join(self.output_dir, "chunk_distribution_chart.png"))
        plt.close()
    
    def _generate_plotly_precision_chart(self, return_fig=False):
        """Generate and save interactive precision chart with Plotly."""
        df = pd.DataFrame({
            "Strategy": [],
            "Precision": [],
            "Efficiency": []
        })
        
        for strategy, metrics in self.aggregated.items():
            # Get metrics with fallbacks
            precision = metrics.get("context_precision", metrics.get("avg_context_precision", 0))
            efficiency = metrics.get("token_efficiency", metrics.get("avg_token_efficiency", 0))
            
            df = pd.concat([df, pd.DataFrame({
                "Strategy": [strategy],
                "Precision": [precision],
                "Efficiency": [efficiency]
            })], ignore_index=True)
        
        # Color scheme
        colors = {
            self.best_strategy: '#4cc9f0',  # Special color for best strategy
            'precision': '#4361ee',
            'efficiency': '#3a0ca3'
        }
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Context Precision", "Token Efficiency"))
        
        # Add precision bars
        fig.add_trace(
            go.Bar(
                x=df["Strategy"],
                y=df["Precision"],
                name="Precision",
                marker_color=[colors[self.best_strategy] if s == self.best_strategy else colors['precision'] for s in df["Strategy"]],
                hovertemplate='<b>%{x}</b><br>Precision: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add efficiency bars
        fig.add_trace(
            go.Bar(
                x=df["Strategy"],
                y=df["Efficiency"],
                name="Efficiency",
                marker_color=[colors[self.best_strategy] if s == self.best_strategy else colors['efficiency'] for s in df["Strategy"]],
                hovertemplate='<b>%{x}</b><br>Efficiency: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': "Precision and Efficiency by Chunking Strategy",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, family='Inter, sans-serif')
            },
            height=500,
            width=900,
            showlegend=False,
            font=dict(family='Inter, sans-serif'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Set y-axis range and grid
        fig.update_yaxes(
            range=[0, 1], 
            gridcolor='#f0f0f0',
            tickfont=dict(size=10)
        )
        
        # Set x-axis properties
        fig.update_xaxes(
            tickangle=30,
            tickfont=dict(size=10)
        )
        
        if return_fig:
            return fig
        
        # Save the chart
        if "html" in self.formats:
            fig.write_html(os.path.join(self.output_dir, "precision_efficiency_plotly.html"))
        if "png" in self.formats:
            fig.write_image(os.path.join(self.output_dir, "precision_efficiency_plotly.png"))
    
    def _generate_plotly_token_efficiency_chart(self, return_fig=False):
        """Generate and save interactive token efficiency chart with Plotly."""
        df = pd.DataFrame({
            "Strategy": [],
            "Context Tokens": []
        })
        
        for strategy, metrics in self.aggregated.items():
            df = pd.concat([df, pd.DataFrame({
                "Strategy": [strategy],
                "Context Tokens": [metrics.get("avg_context_tokens", 0)]
            })], ignore_index=True)
        
        fig = px.bar(
            df,
            x="Strategy",
            y="Context Tokens",
            title="Average Context Token Usage by Chunking Strategy",
            color="Strategy",
            color_discrete_map={s: 'green' if s == self.best_strategy else 'orange' for s in df["Strategy"]}
        )
        
        fig.update_layout(
            height=500,
            width=800,
            showlegend=False
        )
        
        if return_fig:
            return fig
        
        # Save the chart
        if "html" in self.formats:
            fig.write_html(os.path.join(self.output_dir, "token_usage_plotly.html"))
        if "png" in self.formats:
            fig.write_image(os.path.join(self.output_dir, "token_usage_plotly.png"))
    
    def _generate_plotly_time_chart(self, return_fig=False):
        """Generate and save interactive time comparison chart with Plotly."""
        df = pd.DataFrame({
            "Strategy": [],
            "Type": [],
            "Time": [],
            "Tokens": []
        })
        
        for strategy, metrics in self.aggregated.items():
            # Get time metrics with fallbacks
            retrieval_time = metrics.get("retrieval_time", metrics.get("avg_retrieval_time", 0))
            generation_time = metrics.get("generation_time", metrics.get("avg_generation_time", 0))
            
            # If we have latency but not detailed times, use latency as retrieval time
            if retrieval_time == 0 and generation_time == 0:
                latency = metrics.get("latency", metrics.get("avg_latency", metrics.get("avg_total_time", 0)))
                retrieval_time = latency
            
            # Get token count with fallback
            tokens = metrics.get("context_tokens", metrics.get("avg_context_tokens", 0))
            
            df = pd.concat([df, pd.DataFrame({
                "Strategy": [strategy, strategy],
                "Type": ["Retrieval", "Generation"],
                "Time": [retrieval_time, generation_time],
                "Tokens": [tokens] * 2
            })], ignore_index=True)
        
        # Color scheme
        colors = {
            self.best_strategy: '#4cc9f0',  # Special color for best strategy
            'retrieval': '#4361ee',
            'generation': '#3a0ca3',
            'tokens': '#f72585'
        }
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Processing Time", "Context Tokens"),
                          specs=[[{"type": "bar"}, {"type": "bar"}]])
        
        # Add time bars - only show types that have non-zero values
        for time_type, color_key in [("Retrieval", 'retrieval'), ("Generation", 'generation')]:
            subset = df[df["Type"] == time_type]
            if subset["Time"].sum() > 0:  # Only show if we have non-zero times
                fig.add_trace(
                    go.Bar(
                        x=subset["Strategy"],
                        y=subset["Time"],
                        name=time_type,
                        marker_color=colors[color_key],
                        hovertemplate='<b>%{x}</b><br>' + time_type + ' Time: %{y:.2f}s<extra></extra>',
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add token bars
        token_df = df.drop_duplicates("Strategy")
        fig.add_trace(
            go.Bar(
                x=token_df["Strategy"],
                y=token_df["Tokens"],
                name="Tokens",
                marker_color=[colors[self.best_strategy] if s == self.best_strategy else colors['tokens'] for s in token_df["Strategy"]],
                hovertemplate='<b>%{x}</b><br>Tokens: %{y}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={
                'text': "Processing Time and Token Usage by Chunking Strategy",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, family='Inter, sans-serif')
            },
            height=500,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family='Inter, sans-serif', size=10)
            ),
            font=dict(family='Inter, sans-serif'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Update axes
        fig.update_xaxes(
            tickangle=30,
            tickfont=dict(size=10),
            gridcolor='#f0f0f0'
        )
        
        fig.update_yaxes(
            gridcolor='#f0f0f0',
            tickfont=dict(size=10)
        )
        
        if return_fig:
            return fig
        
        # Save the chart
        if "html" in self.formats:
            fig.write_html(os.path.join(self.output_dir, "time_tokens_plotly.html"))
        if "png" in self.formats:
            fig.write_image(os.path.join(self.output_dir, "time_tokens_plotly.png"))
    
    def _generate_plotly_radar_chart(self, return_fig=False):
        """Generate and save interactive radar chart with Plotly."""
        # Prepare data
        strategies = list(self.aggregated.keys())
        
        # Debug: print out available metrics for first strategy
        if strategies and self.debug:
            print(f"Available metrics for {strategies[0]}: {self.aggregated[strategies[0]].keys()}")
        
        # Define the metrics we want to show and create a normalized metrics dict
        metric_mapping = {
            "context_precision": ["context_precision", "avg_context_precision"],
            "token_efficiency": ["token_efficiency", "avg_token_efficiency"],
            "answer_relevance": ["answer_relevance", "avg_answer_relevance"], 
            "chunk_similarities": ["chunk_similarities", "avg_chunk_similarities"]
        }
        
        # Create a normalized metrics dictionary
        normalized_data = {}
        for strategy in strategies:
            normalized_data[strategy] = {}
            # Extract metrics with fallbacks
            for metric_key, possible_keys in metric_mapping.items():
                for key in possible_keys:
                    if key in self.aggregated[strategy]:
                        normalized_data[strategy][metric_key] = self.aggregated[strategy][key]
                        break
                else:
                    # Default to 0 if no matching key found
                    normalized_data[strategy][metric_key] = 0
        
        # Handle latency/time metrics
        for strategy in strategies:
            # Find time metric with fallbacks
            time_value = 0
            for key in ["latency", "avg_latency", "avg_total_time", "total_time"]:
                if key in self.aggregated[strategy]:
                    time_value = self.aggregated[strategy][key]
                    break
            
            # Store original time value
            normalized_data[strategy]["time_original"] = time_value
        
        # Normalize time values (lower is better)
        max_time = max([data["time_original"] for strategy, data in normalized_data.items()])
        if max_time > 0:
            for strategy in strategies:
                normalized_data[strategy]["time_efficiency"] = 1 - (normalized_data[strategy]["time_original"] / max_time)
        else:
            for strategy in strategies:
                normalized_data[strategy]["time_efficiency"] = 0
        
        # Define the metrics to display on the radar chart
        display_metrics = [
            ("context_precision", "Context Precision"),
            ("token_efficiency", "Token Efficiency"),
            ("chunk_similarities", "Chunk Similarities"),
            ("time_efficiency", "Time Efficiency")
        ]
        
        # If answer relevance has non-zero values, include it
        if any(data["answer_relevance"] > 0 for strategy, data in normalized_data.items()):
            display_metrics.insert(2, ("answer_relevance", "Answer Relevance"))
        
        # Extract metrics and their friendly names
        metric_keys = [m[0] for m in display_metrics]
        metric_names = [m[1] for m in display_metrics]
        
        # Color scheme
        colors = {
            self.best_strategy: '#4cc9f0',  # Highlight color for best strategy
            'default': ['#4361ee', '#3a0ca3', '#7209b7', '#f72585', '#4cc9f0']  # Colors for other strategies
        }
        
        # Create the figure
        fig = go.Figure()
        
        # Add traces for each strategy
        for i, strategy in enumerate(strategies):
            # Get values for this strategy
            values = [normalized_data[strategy].get(metric, 0) for metric in metric_keys]
            
            # Close the loop by adding the first value at the end
            values.append(values[0])
            radar_metrics = metric_names + [metric_names[0]]
            
            # Choose color (special for best strategy)
            if strategy == self.best_strategy:
                color = colors[self.best_strategy]
                line_width = 3
            else:
                color = colors['default'][i % len(colors['default'])]
                line_width = 2
            
            # Convert hex color to rgba for fill with transparency
            # Extract hex color without # and convert to RGB
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)'
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics,
                fill='toself',
                name=strategy,
                line_color=color,
                line_width=line_width,
                fillcolor=fill_color
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10),
                    tickvals=[0.2, 0.4, 0.6, 0.8],
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, family='Inter, sans-serif'),
                )
            ),
            title={
                'text': "Strategy Performance Comparison",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, family='Inter, sans-serif')
            },
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                font=dict(family='Inter, sans-serif')
            ),
            margin=dict(l=80, r=80, t=80, b=80),
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        
        if return_fig:
            return fig
        
        # Save the chart
        if "html" in self.formats:
            fig.write_html(os.path.join(self.output_dir, "radar_plotly.html"))
        if "png" in self.formats:
            fig.write_image(os.path.join(self.output_dir, "radar_plotly.png"))
    
    def _generate_plotly_combined_metrics(self, return_fig=False):
        """Generate and save interactive combined metrics chart with Plotly."""
        # Prepare data
        df = pd.DataFrame(columns=["Strategy", "Metric", "Value"])
        
        for strategy, metrics in self.aggregated.items():
            strategy_data = {
                "Strategy": [strategy] * 5,
                "Metric": [
                    "Context Precision",
                    "Token Efficiency",
                    "Chunk Similarities",
                    "Time Efficiency",
                    "Context Tokens (normalized)"
                ],
                "Value": [
                    metrics.get("avg_context_precision", 0),
                    metrics.get("avg_token_efficiency", 0),
                    metrics.get("avg_chunk_similarities", 0),
                    # Normalize time (lower is better)
                    1 - min(1, metrics.get("avg_total_time", 0) / 5),  # Assume 5 seconds is max reasonable time
                    # Normalize tokens (lower is better for efficiency, but we need some tokens)
                    min(1, metrics.get("avg_context_tokens", 0) / 1000)  # Normalize to 1000 tokens
                ]
            }
            df = pd.concat([df, pd.DataFrame(strategy_data)], ignore_index=True)
        
        # Create the figure
        fig = px.bar(
            df,
            x="Strategy",
            y="Value",
            color="Metric",
            barmode="group",
            title="Combined Performance Metrics by Chunking Strategy",
            labels={"Value": "Normalized Score (0-1)"}
        )
        
        # Highlight the best strategy
        if self.best_strategy:
            fig.add_shape(
                type="rect",
                x0=df[df["Strategy"] == self.best_strategy]["Strategy"].iloc[0],
                x1=df[df["Strategy"] == self.best_strategy]["Strategy"].iloc[0],
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor="rgba(0,255,0,0.1)",
                layer="below",
                line_width=0,
            )
        
        fig.update_layout(
            height=600,
            width=1000
        )
        
        if return_fig:
            return fig
        
        # Save the chart
        if "html" in self.formats:
            fig.write_html(os.path.join(self.output_dir, "combined_metrics_plotly.html"))
        if "png" in self.formats:
            fig.write_image(os.path.join(self.output_dir, "combined_metrics_plotly.png")) 