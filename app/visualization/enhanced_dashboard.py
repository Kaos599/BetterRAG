from typing import Dict, List, Any
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from dash import Dash, html, dcc, Input, Output, dash_table, State, callback_context
import dash_bootstrap_components as dbc


class ParameterOptimizationDashboard:
    """
    Dashboard for visualizing the impact of different parameter values on RAG performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dashboard with configuration.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.output_dir = config.get("general", {}).get("output_directory", "./output/")
        self.dashboard_config = config.get("dashboard", {})
        self.port = self.dashboard_config.get("port", 8051)  # Different port from main dashboard
        self.debug = self.dashboard_config.get("debug", False)
        
        # Parameters to analyze
        self.parameters = {
            "fixed_size": ["chunk_size", "chunk_overlap", "overlap_percentage"],
            "recursive": ["chunk_size", "chunk_overlap", "overlap_percentage", "separator_set"],
            "semantic": ["similarity_threshold", "min_chunk_size", "max_chunk_size", "semantic_method"]
        }
        
        # Metrics to show
        self.metrics = ["context_precision", "token_efficiency", "latency", "combined_score"]
        
        # Store for results
        self.results = None
        self.aggregated = None
        self.best_strategy = None
    
    def save_results(self, evaluation_results: Dict[str, Any], aggregated_metrics: Dict[str, Any], best_strategy: str):
        """
        Save the parameter optimization results to a JSON file.
        
        Args:
            evaluation_results: Detailed evaluation results for each strategy
            aggregated_metrics: Aggregated metrics for each strategy
            best_strategy: Name of the best performing strategy
        """
        self.results = evaluation_results
        self.aggregated = aggregated_metrics
        self.best_strategy = best_strategy
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save results to file
        results_file = os.path.join(self.output_dir, "parameter_optimization_results.json")
        
        data = {
            "evaluation_results": evaluation_results,
            "aggregated_metrics": aggregated_metrics,
            "best_strategy": best_strategy
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Parameter optimization results saved to {results_file}")
            return True
        except Exception as e:
            print(f"Error saving parameter optimization results: {e}")
            return False
    
    def load_results(self, results_file: str = None):
        """
        Load evaluation results from a JSON file.
        
        Args:
            results_file: Path to results file (defaults to evaluation_results.json in output_dir)
        """
        if results_file is None:
            results_file = os.path.join(self.output_dir, "parameter_optimization_results.json")
        
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            return False
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                self.results = data.get("evaluation_results", {})
                self.aggregated = data.get("aggregated_metrics", {})
                self.best_strategy = data.get("best_strategy", "")
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def prepare_parameter_analysis_data(self):
        """
        Extract parameter values and corresponding metrics for analysis.
        
        Returns:
            DataFrame with parameters and metrics for each strategy
        """
        if not self.aggregated:
            return pd.DataFrame()
        
        rows = []
        
        for strategy, metrics in self.aggregated.items():
            # Extract chunking strategy type and parameters
            strategy_type = None
            params = {}
            
            if "fixed_size" in strategy:
                strategy_type = "fixed_size"
                # Extract chunk_size and overlap from strategy name (fixed_size_512_15pct)
                parts = strategy.split('_')
                if len(parts) >= 3:
                    chunk_size = int(parts[2])
                    overlap_pct = int(parts[3].replace('pct', ''))
                    params["chunk_size"] = chunk_size
                    params["chunk_overlap"] = int(chunk_size * overlap_pct / 100)
                    params["overlap_percentage"] = overlap_pct
            
            elif "recursive" in strategy:
                strategy_type = "recursive"
                # Extract parameters from strategy name (recursive_512_15pct_sep1)
                parts = strategy.split('_')
                if len(parts) >= 4:
                    chunk_size = int(parts[1])
                    overlap_pct = int(parts[2].replace('pct', ''))
                    separator_set = parts[3]
                    params["chunk_size"] = chunk_size
                    params["chunk_overlap"] = int(chunk_size * overlap_pct / 100)
                    params["overlap_percentage"] = overlap_pct
                    params["separator_set"] = separator_set
            
            elif "semantic" in strategy:
                strategy_type = "semantic"
                # Extract parameters from strategy name (semantic_standard_0.75_100_600)
                parts = strategy.split('_')
                if len(parts) >= 5:
                    method = parts[1]
                    threshold = float(parts[2])
                    min_size = int(parts[3])
                    max_size = int(parts[4])
                    params["semantic_method"] = method
                    params["similarity_threshold"] = threshold
                    params["min_chunk_size"] = min_size
                    params["max_chunk_size"] = max_size
            
            if strategy_type:
                # Extract metrics
                row = {
                    "strategy_name": strategy,
                    "strategy_type": strategy_type,
                    "context_precision": metrics.get("context_precision", 0),
                    "token_efficiency": metrics.get("token_efficiency", 0),
                    "latency": metrics.get("latency", 0),
                    "combined_score": metrics.get("combined_score", 0)
                }
                
                # Add parameters
                row.update(params)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def run_dashboard(self):
        """
        Run the parameter optimization dashboard.
        """
        if not self.load_results():
            print("No results to display. Please run evaluation first.")
            return
        
        # Prepare analysis data
        df = self.prepare_parameter_analysis_data()
        
        if df.empty:
            print("No parameter data available for analysis.")
            return
        
        # Create app
        app = Dash(__name__, 
                  external_stylesheets=[dbc.themes.BOOTSTRAP],
                  title="RAG Parameter Optimization Dashboard")
        
        # Create controls
        strategy_options = [
            {"label": "Fixed Size Chunking", "value": "fixed_size"},
            {"label": "Recursive Chunking", "value": "recursive"},
            {"label": "Semantic Chunking", "value": "semantic"}
        ]
        
        parameter_options = {
            "fixed_size": [
                {"label": "Chunk Size", "value": "chunk_size"},
                {"label": "Chunk Overlap", "value": "chunk_overlap"},
                {"label": "Overlap Percentage", "value": "overlap_percentage"}
            ],
            "recursive": [
                {"label": "Chunk Size", "value": "chunk_size"},
                {"label": "Chunk Overlap", "value": "chunk_overlap"},
                {"label": "Overlap Percentage", "value": "overlap_percentage"},
                {"label": "Separator Set", "value": "separator_set"}
            ],
            "semantic": [
                {"label": "Similarity Threshold", "value": "similarity_threshold"},
                {"label": "Min Chunk Size", "value": "min_chunk_size"},
                {"label": "Max Chunk Size", "value": "max_chunk_size"},
                {"label": "Semantic Method", "value": "semantic_method"}
            ]
        }
        
        metric_options = [
            {"label": "Context Precision", "value": "context_precision"},
            {"label": "Token Efficiency", "value": "token_efficiency"},
            {"label": "Latency (seconds)", "value": "latency"},
            {"label": "Combined Score", "value": "combined_score"}
        ]
        
        # Layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("RAG Parameter Optimization Dashboard", 
                            className="text-center my-4"),
                    html.P("Explore how different parameter values affect RAG performance metrics.",
                           className="text-center mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Parameter Analysis Controls"),
                        dbc.CardBody([
                            html.H5("1. Select Chunking Strategy Type"),
                            dcc.RadioItems(
                                id="strategy-type-selector",
                                options=strategy_options,
                                value="fixed_size",
                                className="mb-3"
                            ),
                            
                            html.H5("2. Select Parameter to Analyze"),
                            dcc.Dropdown(
                                id="parameter-selector",
                                options=parameter_options["fixed_size"],
                                value="overlap_percentage",
                                className="mb-3"
                            ),
                            
                            html.H5("3. Select Performance Metric"),
                            dcc.Dropdown(
                                id="metric-selector",
                                options=metric_options,
                                value="combined_score",
                                className="mb-3"
                            ),
                            
                            html.H5("4. Filter Options (Optional)"),
                            html.Div(id="filter-controls", className="mb-3")
                        ])
                    ], className="mb-4")
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Parameter Impact Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="parameter-impact-chart", style={"height": "500px"})
                        ])
                    ], className="mb-4")
                ], width=8)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Detailed Performance Comparison"),
                        dbc.CardBody([
                            html.Div([
                                dbc.Tabs([
                                    dbc.Tab([
                                        dcc.Graph(id="parameter-heatmap", style={"height": "400px"})
                                    ], label="Heatmap View"),
                                    dbc.Tab([
                                        dash_table.DataTable(
                                            id="results-table",
                                            columns=[
                                                {"name": "Strategy", "id": "strategy_name"},
                                                {"name": "Type", "id": "strategy_type"},
                                                {"name": "Context Precision", "id": "context_precision", "type": "numeric", "format": {"specifier": ".4f"}},
                                                {"name": "Token Efficiency", "id": "token_efficiency", "type": "numeric", "format": {"specifier": ".4f"}},
                                                {"name": "Latency (s)", "id": "latency", "type": "numeric", "format": {"specifier": ".4f"}},
                                                {"name": "Combined Score", "id": "combined_score", "type": "numeric", "format": {"specifier": ".4f"}}
                                            ],
                                            data=df.to_dict("records"),
                                            filter_action="native",
                                            sort_action="native",
                                            sort_mode="multi",
                                            page_size=10,
                                            style_table={"overflowX": "auto"},
                                            style_data_conditional=[
                                                {
                                                    "if": {"row_index": "odd"},
                                                    "backgroundColor": "rgb(248, 248, 248)"
                                                }
                                            ],
                                            style_header={
                                                "backgroundColor": "rgb(230, 230, 230)",
                                                "fontWeight": "bold"
                                            }
                                        )
                                    ], label="Data Table"),
                                    dbc.Tab([
                                        dcc.Graph(id="parallel-coordinates", style={"height": "500px"})
                                    ], label="Parallel Coordinates")
                                ])
                            ])
                        ])
                    ])
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id="optimal-parameters-card", className="mt-4")
                ])
            ]),
            
            # Store the DataFrame in browser
            dcc.Store(id="analysis-data", data=df.to_dict("records"))
            
        ], fluid=True)
        
        # Callbacks
        @app.callback(
            Output("parameter-selector", "options"),
            Output("parameter-selector", "value"),
            Input("strategy-type-selector", "value")
        )
        def update_parameter_options(strategy_type):
            """Update parameter dropdown based on selected strategy type."""
            options = parameter_options.get(strategy_type, [])
            default_value = options[0]["value"] if options else None
            return options, default_value
        
        @app.callback(
            Output("filter-controls", "children"),
            Input("strategy-type-selector", "value"),
            Input("parameter-selector", "value")
        )
        def update_filter_controls(strategy_type, parameter):
            """Update filter controls based on selected strategy and parameter."""
            # Filter the dataframe for the selected strategy type
            strategy_df = df[df["strategy_type"] == strategy_type]
            
            if parameter not in strategy_df.columns:
                return html.Div("No filter options available")
            
            # Get unique values for the parameter
            if strategy_df[parameter].dtype in [np.int64, np.float64]:
                # For numeric parameters, create a range slider
                min_val = strategy_df[parameter].min()
                max_val = strategy_df[parameter].max()
                step = 1 if strategy_df[parameter].dtype == np.int64 else 0.05
                
                return html.Div([
                    html.Label(f"Filter by {parameter} range:"),
                    dcc.RangeSlider(
                        id="parameter-filter",
                        min=min_val,
                        max=max_val,
                        step=step,
                        marks={i: str(i) for i in np.linspace(min_val, max_val, 5).round(2)},
                        value=[min_val, max_val]
                    )
                ])
            else:
                # For categorical parameters, create a checklist
                options = [{"label": str(val), "value": str(val)} for val in strategy_df[parameter].unique()]
                
                return html.Div([
                    html.Label(f"Filter by {parameter}:"),
                    dcc.Checklist(
                        id="parameter-filter",
                        options=options,
                        value=[opt["value"] for opt in options],
                        inline=True
                    )
                ])
        
        @app.callback(
            Output("parameter-impact-chart", "figure"),
            Input("strategy-type-selector", "value"),
            Input("parameter-selector", "value"),
            Input("metric-selector", "value"),
            Input("parameter-filter", "value"),
            State("analysis-data", "data")
        )
        def update_parameter_chart(strategy_type, parameter, metric, filter_value, data):
            """Update the parameter impact chart based on selections."""
            # Convert stored data back to DataFrame
            chart_df = pd.DataFrame(data)
            
            # Filter by strategy type
            chart_df = chart_df[chart_df["strategy_type"] == strategy_type]
            
            # Apply filter if available
            if filter_value:
                if isinstance(filter_value, list) and len(filter_value) == 2 and all(isinstance(v, (int, float)) for v in filter_value):
                    # Range filter for numeric values
                    chart_df = chart_df[(chart_df[parameter] >= filter_value[0]) & 
                                       (chart_df[parameter] <= filter_value[1])]
                elif isinstance(filter_value, list) and all(isinstance(v, str) for v in filter_value):
                    # Categorical filter
                    chart_df = chart_df[chart_df[parameter].astype(str).isin(filter_value)]
            
            # If the parameter is not in the dataframe, return empty figure
            if parameter not in chart_df.columns or metric not in chart_df.columns:
                return go.Figure().update_layout(
                    title="No data available for the selected parameters",
                    xaxis_title="N/A",
                    yaxis_title="N/A"
                )
            
            # Prepare data for chart
            if chart_df[parameter].dtype in [np.int64, np.float64]:
                # For numeric parameters, create a scatter plot with trend line
                fig = px.scatter(
                    chart_df,
                    x=parameter,
                    y=metric,
                    color="strategy_name",
                    hover_data=["strategy_name"] + [p for p in self.parameters[strategy_type] if p in chart_df.columns],
                    trendline="ols",
                    title=f"Impact of {parameter} on {metric}",
                    labels={parameter: parameter.replace("_", " ").title(), 
                           metric: metric.replace("_", " ").title()}
                )
                
                # Add best fit line
                try:
                    from scipy import stats
                    x = chart_df[parameter]
                    y = chart_df[metric]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Add annotation with correlation statistics
                    fig.add_annotation(
                        x=0.05,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"R² = {r_value**2:.4f}<br>p-value = {p_value:.4f}",
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                except:
                    pass
            else:
                # For categorical parameters, create a box plot
                fig = px.box(
                    chart_df,
                    x=parameter,
                    y=metric,
                    color=parameter,
                    title=f"Impact of {parameter} on {metric}",
                    labels={parameter: parameter.replace("_", " ").title(), 
                           metric: metric.replace("_", " ").title()}
                )
                
                # Add individual points
                fig.add_traces(
                    px.strip(
                        chart_df,
                        x=parameter,
                        y=metric,
                        color=parameter
                    ).data
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title=parameter.replace("_", " ").title(),
                yaxis_title=metric.replace("_", " ").title(),
                legend_title="Strategy",
                font=dict(family="Arial", size=12),
                hovermode="closest"
            )
            
            return fig
        
        @app.callback(
            Output("parameter-heatmap", "figure"),
            Input("strategy-type-selector", "value"),
            Input("metric-selector", "value"),
            State("analysis-data", "data")
        )
        def update_heatmap(strategy_type, metric, data):
            """Update the parameter heatmap visualization."""
            heatmap_df = pd.DataFrame(data)
            
            # Filter by strategy type
            heatmap_df = heatmap_df[heatmap_df["strategy_type"] == strategy_type]
            
            if heatmap_df.empty:
                return go.Figure().update_layout(title="No data available for heatmap")
            
            # Get parameters for the strategy type
            x_param = None
            y_param = None
            
            if strategy_type == "fixed_size":
                x_param = "chunk_size"
                y_param = "overlap_percentage"
            elif strategy_type == "recursive":
                x_param = "chunk_size"
                y_param = "overlap_percentage"
            elif strategy_type == "semantic":
                x_param = "similarity_threshold"
                y_param = "min_chunk_size"
            
            if not x_param or not y_param or x_param not in heatmap_df.columns or y_param not in heatmap_df.columns:
                return go.Figure().update_layout(title="Could not determine parameters for heatmap")
            
            # Create pivot table for heatmap
            pivot_df = heatmap_df.pivot_table(
                values=metric,
                index=y_param,
                columns=x_param,
                aggfunc='mean'
            )
            
            # Create heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(
                    x=x_param.replace("_", " ").title(),
                    y=y_param.replace("_", " ").title(),
                    color=metric.replace("_", " ").title()
                ),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="Viridis",
                title=f"Heatmap of {metric.replace('_', ' ').title()} by {x_param.replace('_', ' ').title()} and {y_param.replace('_', ' ').title()}"
            )
            
            # Update layout
            fig.update_layout(
                font=dict(family="Arial", size=12),
                coloraxis_colorbar=dict(
                    title=metric.replace("_", " ").title()
                )
            )
            
            return fig
        
        @app.callback(
            Output("parallel-coordinates", "figure"),
            Input("strategy-type-selector", "value"),
            State("analysis-data", "data")
        )
        def update_parallel_coordinates(strategy_type, data):
            """Update the parallel coordinates visualization."""
            pc_df = pd.DataFrame(data)
            
            # Filter by strategy type
            pc_df = pc_df[pc_df["strategy_type"] == strategy_type]
            
            if pc_df.empty:
                return go.Figure().update_layout(title="No data available for parallel coordinates")
            
            # Get parameters for this strategy type
            params = [p for p in self.parameters[strategy_type] if p in pc_df.columns]
            metrics = [m for m in self.metrics if m in pc_df.columns]
            
            if not params or not metrics:
                return go.Figure().update_layout(title="No parameters or metrics available")
            
            # Create dimensions list for parallel coordinates
            dimensions = []
            
            # Add parameters
            for param in params:
                dimensions.append(
                    dict(
                        range=[pc_df[param].min(), pc_df[param].max()],
                        label=param.replace("_", " ").title(),
                        values=pc_df[param]
                    )
                )
            
            # Add metrics
            for metric in metrics:
                if metric == "latency":
                    # For latency, lower is better
                    dimensions.append(
                        dict(
                            range=[pc_df[metric].max(), pc_df[metric].min()],  # Invert range so lower is better
                            label=metric.replace("_", " ").title(),
                            values=pc_df[metric]
                        )
                    )
                else:
                    dimensions.append(
                        dict(
                            range=[pc_df[metric].min(), pc_df[metric].max()],
                            label=metric.replace("_", " ").title(),
                            values=pc_df[metric]
                        )
                    )
            
            # Create figure
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=pc_df["combined_score"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Combined Score")
                    ),
                    dimensions=dimensions
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Parallel Coordinates Plot for {strategy_type.replace('_', ' ').title()} Chunking",
                font=dict(family="Arial", size=12)
            )
            
            return fig
        
        @app.callback(
            Output("optimal-parameters-card", "children"),
            Input("strategy-type-selector", "value"),
            Input("metric-selector", "value"),
            State("analysis-data", "data")
        )
        def update_optimal_parameters(strategy_type, metric, data):
            """Update the optimal parameters card."""
            opt_df = pd.DataFrame(data)
            
            # Filter by strategy type
            opt_df = opt_df[opt_df["strategy_type"] == strategy_type]
            
            if opt_df.empty or metric not in opt_df.columns:
                return html.Div("No data available for optimal parameters")
            
            # Find the optimal strategy based on the metric
            if metric == "latency":
                # For latency, lower is better
                best_idx = opt_df[metric].idxmin()
                best_direction = "lowest"
            else:
                # For other metrics, higher is better
                best_idx = opt_df[metric].idxmax()
                best_direction = "highest"
            
            best_strategy = opt_df.loc[best_idx]
            
            # Get parameters for this strategy type
            params = [p for p in self.parameters[strategy_type] if p in opt_df.columns]
            
            # Create parameter value strings
            param_values = []
            for param in params:
                if param in best_strategy:
                    param_values.append(f"{param.replace('_', ' ').title()}: {best_strategy[param]}")
            
            # Create the card
            return dbc.Card([
                dbc.CardHeader(f"Optimal Parameters for {strategy_type.replace('_', ' ').title()} Chunking"),
                dbc.CardBody([
                    html.H5(f"Best Configuration for {metric.replace('_', ' ').title()}"),
                    html.P(f"The following configuration achieved the {best_direction} {metric.replace('_', ' ').title()} value of {best_strategy[metric]:.4f}:"),
                    html.Ul([html.Li(param) for param in param_values]),
                    html.P(f"Strategy name: {best_strategy['strategy_name']}")
                ])
            ])
        
        # Run the app
        app.run_server(debug=self.debug, port=self.port) 