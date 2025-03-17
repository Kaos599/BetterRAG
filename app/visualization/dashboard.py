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
    
    def run_dashboard(self):
        """
        Run an interactive Dash dashboard with the evaluation results.
        """
        if not self.results or not self.aggregated:
            print("No results to display in dashboard.")
            return
        
        # Create app
        app = Dash(__name__)
        
        # Convert aggregated metrics to DataFrame for easier handling
        strategies = list(self.aggregated.keys())
        metrics_df = pd.DataFrame(self.aggregated).T.reset_index()
        metrics_df = metrics_df.rename(columns={"index": "strategy"})
        
        # Create dropdown options for queries
        queries = list(self.results.keys())
        query_options = [{"label": q[:50] + "..." if len(q) > 50 else q, "value": q} for q in queries]
        
        # Layout
        app.layout = html.Div([
            html.H1("Chunking Strategy Evaluation Dashboard"),
            
            html.Div([
                html.H2("Best Strategy"),
                html.Div([
                    html.H3(self.best_strategy),
                    html.P(f"Based on the evaluation, the '{self.best_strategy}' strategy performed best overall.")
                ], style={"border": "2px solid green", "padding": "10px", "border-radius": "5px"})
            ]),
            
            html.Hr(),
            
            html.H2("Aggregated Performance Metrics"),
            
            # Strategy comparison tabs
            dcc.Tabs([
                dcc.Tab(label="Metrics Table", children=[
                    dash_table.DataTable(
                        id="metrics-table",
                        columns=[
                            {"name": "Strategy", "id": "strategy"},
                            {"name": "Context Precision", "id": "avg_context_precision"},
                            {"name": "Token Efficiency", "id": "avg_token_efficiency"},
                            {"name": "Avg Similarity", "id": "avg_chunk_similarities"},
                            {"name": "Total Time (s)", "id": "avg_total_time"},
                            {"name": "Context Tokens", "id": "avg_context_tokens"},
                        ],
                        data=metrics_df.to_dict("records"),
                        style_data_conditional=[
                            {
                                "if": {"row_index": metrics_df[metrics_df["strategy"] == self.best_strategy].index[0]},
                                "backgroundColor": "rgba(0, 255, 0, 0.2)",
                                "fontWeight": "bold"
                            }
                        ],
                        sort_action="native",
                        sort_mode="multi",
                        style_table={"overflowX": "auto"}
                    )
                ]),
                
                dcc.Tab(label="Precision & Efficiency", children=[
                    dcc.Graph(
                        id="precision-efficiency-chart",
                        figure=self._generate_plotly_precision_chart(return_fig=True)
                    )
                ]),
                
                dcc.Tab(label="Time & Tokens", children=[
                    dcc.Graph(
                        id="time-tokens-chart",
                        figure=self._generate_plotly_time_chart(return_fig=True)
                    )
                ]),
                
                dcc.Tab(label="Overall Comparison", children=[
                    dcc.Graph(
                        id="radar-chart",
                        figure=self._generate_plotly_radar_chart(return_fig=True)
                    )
                ]),
            ]),
            
            html.Hr(),
            
            html.H2("Query-Specific Results"),
            
            html.Div([
                html.Label("Select Query:"),
                dcc.Dropdown(
                    id="query-dropdown",
                    options=query_options,
                    value=queries[0] if queries else None
                )
            ]),
            
            html.Div([
                dcc.Graph(id="query-metrics-chart")
            ]),
            
            html.Div([
                html.H3("Retrieved Chunks"),
                html.Div(id="chunks-container")
            ]) if self.show_chunks else html.Div(),
            
            html.Hr(),
            
            html.Div([
                html.H3("Generated Answers"),
                html.Div(id="answers-container")
            ])
        ], style={"margin": "20px", "maxWidth": "1200px"})
        
        # Callbacks
        @app.callback(
            [Output("query-metrics-chart", "figure"),
             Output("chunks-container", "children"),
             Output("answers-container", "children")],
            [Input("query-dropdown", "value")]
        )
        def update_query_results(query):
            if not query or query not in self.results:
                return {}, [], []
            
            # Generate chart for this query's metrics
            query_results = self.results[query]
            
            # Prepare data for the chart
            data = []
            for strategy, metrics in query_results.items():
                if "error" not in metrics:
                    data.append({
                        "strategy": strategy,
                        "retrieval_time": metrics.get("retrieval_time", 0),
                        "context_precision": metrics.get("context_precision", 0),
                        "token_efficiency": metrics.get("token_efficiency", 0),
                        "chunk_similarities_avg": np.mean(metrics.get("chunk_similarities", [0])),
                        "context_tokens": metrics.get("context_tokens", 0)
                    })
            
            # Create chart
            if not data:
                return {}, [], []
            
            df = pd.DataFrame(data)
            
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                subplot_titles=("Context Precision", "Token Efficiency", 
                               "Retrieval Time (s)", "Context Tokens")
            )
            
            # Add bars for each metric
            fig.add_trace(go.Bar(x=df["strategy"], y=df["context_precision"], name="Context Precision"),
                         row=1, col=1)
            fig.add_trace(go.Bar(x=df["strategy"], y=df["token_efficiency"], name="Token Efficiency"),
                         row=1, col=2)
            fig.add_trace(go.Bar(x=df["strategy"], y=df["retrieval_time"], name="Retrieval Time"),
                         row=2, col=1)
            fig.add_trace(go.Bar(x=df["strategy"], y=df["context_tokens"], name="Context Tokens"),
                         row=2, col=2)
            
            fig.update_layout(
                height=600,
                title_text=f"Performance Metrics for Query: {query[:50]}...",
                showlegend=False
            )
            
            # Generate chunks display
            chunks_display = []
            for strategy, metrics in query_results.items():
                if "error" not in metrics and "chunks" in metrics:
                    chunks = metrics["chunks"]
                    
                    chunks_display.append(html.Div([
                        html.H4(f"Strategy: {strategy}"),
                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Chunk"),
                                    html.Th("Similarity"),
                                    html.Th("Text Preview")
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(chunk.get("chunk_id", "")),
                                    html.Td(f"{chunk.get('similarity', 0):.4f}"),
                                    html.Td(chunk.get("text", ""))
                                ]) for chunk in chunks
                            ])
                        ], style={"width": "100%", "border": "1px solid #ddd", "borderCollapse": "collapse"})
                    ], style={"marginBottom": "20px"}))
            
            # Generate answers display
            answers_display = []
            for strategy, metrics in query_results.items():
                if "error" not in metrics and "answer" in metrics:
                    answer = metrics["answer"]
                    
                    answers_display.append(html.Div([
                        html.H4(f"Strategy: {strategy}"),
                        html.Div(
                            html.P(answer),
                            style={"border": "1px solid #ddd", "padding": "10px", "borderRadius": "5px"}
                        )
                    ], style={"marginBottom": "20px"}))
            
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
            df = pd.concat([df, pd.DataFrame({
                "Strategy": [strategy],
                "Precision": [metrics.get("avg_context_precision", 0)],
                "Efficiency": [metrics.get("avg_token_efficiency", 0)]
            })], ignore_index=True)
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Context Precision", "Token Efficiency"))
        
        # Add precision bars
        fig.add_trace(
            go.Bar(
                x=df["Strategy"],
                y=df["Precision"],
                name="Precision",
                marker_color=['green' if s == self.best_strategy else 'royalblue' for s in df["Strategy"]]
            ),
            row=1, col=1
        )
        
        # Add efficiency bars
        fig.add_trace(
            go.Bar(
                x=df["Strategy"],
                y=df["Efficiency"],
                name="Efficiency",
                marker_color=['green' if s == self.best_strategy else 'lightgreen' for s in df["Strategy"]]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Precision and Efficiency by Chunking Strategy",
            height=500,
            width=900,
            showlegend=False
        )
        
        # Set y-axis range
        fig.update_yaxes(range=[0, 1])
        
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
            df = pd.concat([df, pd.DataFrame({
                "Strategy": [strategy, strategy],
                "Type": ["Retrieval", "Generation"],
                "Time": [
                    metrics.get("avg_retrieval_time", 0),
                    metrics.get("avg_generation_time", 0)
                ],
                "Tokens": [metrics.get("avg_context_tokens", 0)] * 2
            })], ignore_index=True)
        
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("Processing Time", "Context Tokens"),
                          specs=[[{"type": "bar"}, {"type": "bar"}]])
        
        # Add time bars
        for i, time_type in enumerate(["Retrieval", "Generation"]):
            subset = df[df["Type"] == time_type]
            fig.add_trace(
                go.Bar(
                    x=subset["Strategy"],
                    y=subset["Time"],
                    name=time_type,
                    marker_color='coral' if time_type == "Retrieval" else 'lightblue',
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
                marker_color=['green' if s == self.best_strategy else 'orange' for s in token_df["Strategy"]],
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Processing Time and Token Usage by Chunking Strategy",
            height=500,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
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
        
        # Define the metrics we want to show
        metrics = [
            "avg_context_precision",
            "avg_token_efficiency",
            "avg_chunk_similarities"
        ]
        
        # Friendly names for the metrics
        metric_names = [
            "Context Precision",
            "Token Efficiency",
            "Chunk Similarities"
        ]
        
        # Normalize time (lower is better)
        time_values = [self.aggregated[strategy].get("avg_total_time", 0) for strategy in strategies]
        if any(time_values):
            max_time = max(time_values)
            for strategy in strategies:
                time_val = self.aggregated[strategy].get("avg_total_time", 0)
                self.aggregated[strategy]["normalized_time"] = 1 - (time_val / max_time) if max_time > 0 else 0
            
            metrics.append("normalized_time")
            metric_names.append("Time Efficiency")
        
        # Create the figure
        fig = go.Figure()
        
        for strategy in strategies:
            values = [self.aggregated[strategy].get(metric, 0) for metric in metrics]
            # Close the loop by adding the first value at the end
            values.append(values[0])
            radar_metrics = metric_names + [metric_names[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics,
                fill='toself',
                name=strategy,
                line_color='green' if strategy == self.best_strategy else None
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Strategy Performance Comparison",
            height=600,
            width=800,
            showlegend=True
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