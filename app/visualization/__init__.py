from typing import Dict, Any
from app.visualization.dashboard import VisualizationDashboard
from app.visualization.enhanced_dashboard import ParameterOptimizationDashboard

def get_dashboard(config: Dict[str, Any]):
    """
    Get a visualization dashboard instance.
    
    Args:
        config: Visualization configuration
        
    Returns:
        VisualizationDashboard instance
    """
    return VisualizationDashboard(config)

def get_parameter_dashboard(config: Dict[str, Any]):
    """
    Get a parameter optimization dashboard instance.
    
    Args:
        config: Visualization configuration
        
    Returns:
        ParameterOptimizationDashboard instance
    """
    return ParameterOptimizationDashboard(config)

__all__ = ['VisualizationDashboard', 'ParameterOptimizationDashboard', 'get_dashboard', 'get_parameter_dashboard'] 