import os
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_URL = os.getenv('DATABRICKS_URL')
DATABRICKS_API_TOKEN = os.getenv('DATABRICKS_API_TOKEN')
DATABRICKS_API_UPLOAD_PATH = '/Volumes/workspace_team1/default/documents'
