import mlflow
import requests
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DATABRICKS_URL = os.getenv('DATABRICKS_URL')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_API_TOKEN')
DATABRICKS_MODEL_RAG_ENDPOINT = os.getenv('DATABRICKS_MODEL_RAG_ENDPOINT')
DATABRICKS_MODEL_BASE_ENDPOINT = os.getenv('DATABRICKS_MODEL_BASE_ENDPOINT')
DATABRICKS_CATALOG = os.getenv('DATABRICKS_CATALOG')
DATABRICKS_SCHEMA = os.getenv('DATABRICKS_SCHEMA')
MLFLOW_EXPERIMENT_ID = os.getenv('MLFLOW_EXPERIMENT_ID')
DATABRICKS_API_UPLOAD_PATH = F'/Volumes/{DATABRICKS_CATALOG}/{DATABRICKS_SCHEMA}/documents/landing'

def init():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_ID)
    print('TRACKING_URL', mlflow.get_tracking_uri())
    print('DATABRICKS_URL', DATABRICKS_URL)
    print('DATABRICKS_API_UPLOAD_PATH', DATABRICKS_API_UPLOAD_PATH)
    print('MLFLOW_EXPERIMENT_ID', MLFLOW_EXPERIMENT_ID)

def get_warehouse_id():
    try:
        response = requests.get(
            f"{DATABRICKS_URL}/api/2.0/sql/warehouses",
            headers={'Authorization': f'Bearer {DATABRICKS_TOKEN}'}
        )
        response.raise_for_status()
        warehouses = response.json().get('warehouses', [])
        if warehouses:
            return warehouses[0]['id']
    except Exception as e:
        logger.error(f"Error fetching warehouse ID: {str(e)}")
    return None
