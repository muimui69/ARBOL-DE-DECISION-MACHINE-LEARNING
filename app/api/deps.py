from app.core.database import MongoDB
from typing import Dict, Any
import pandas as pd

async def get_db_data() -> Dict[str, pd.DataFrame]:
    """Dependency to get all the necessary data from MongoDB"""
    return await MongoDB.get_all_data()

