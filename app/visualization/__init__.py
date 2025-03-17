from app.visualization.dashboard import VisualizationDashboard

def get_dashboard(config):
    """
    Get a visualization dashboard instance.
    
    Args:
        config: Visualization configuration
        
    Returns:
        VisualizationDashboard instance
    """
    return VisualizationDashboard(config) 