import requests
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DATABRICKS_URL = os.getenv('DATABRICKS_URL')
DATABRICKS_API_TOKEN = os.getenv('DATABRICKS_API_TOKEN')
DATABRICKS_MODEL_RAG_ENDPOINT = os.getenv('DATABRICKS_MODEL_RAG_ENDPOINT')
DATABRICKS_MODEL_BASE_ENDPOINT = os.getenv('DATABRICKS_MODEL_BASE_ENDPOINT')
DATABRICKS_API_UPLOAD_PATH = '/Volumes/workspace_team1/default/documents/landing'


def get_warehouse_id():
    """Fetch the first available warehouse ID"""
    try:
        response = requests.get(
            f"{DATABRICKS_URL}/api/2.0/sql/warehouses",
            headers={'Authorization': f'Bearer {DATABRICKS_API_TOKEN}'}
        )
        response.raise_for_status()
        warehouses = response.json().get('warehouses', [])
        if warehouses:
            return warehouses[0]['id']
    except Exception as e:
        logger.error(f"Error fetching warehouse ID: {str(e)}")
    return None
