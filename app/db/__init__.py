from app.db.mongodb import MongoDBConnector

def get_db_connector(config):
    """
    Get a database connector based on configuration.
    
    Args:
        config: Database configuration dictionary
        
    Returns:
        Database connector instance
    """
    return MongoDBConnector(config) 