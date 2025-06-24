import requests
import uuid
import config
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def upload(file) -> str:
    _id = str(uuid.uuid4())
    file_extension = file.name.split('.')[-1]

    uuid_filename = f'{_id}.{file_extension}'
    volume_path = f'{config.DATABRICKS_API_UPLOAD_PATH}/{uuid_filename}'

    response = requests.put(
        f"{config.DATABRICKS_URL}/api/2.0/fs/files/{volume_path}",
        headers={
            'Authorization': f'Bearer {config.DATABRICKS_API_TOKEN}',
        },
        data=file.getbuffer(),
    )
    response.raise_for_status()
    return _id

def run_sql(**kwargs):
    try:
        response = requests.post(
            f"{config.DATABRICKS_URL}/api/2.0/sql/statements/",
            headers={
                'Authorization': f'Bearer {config.DATABRICKS_API_TOKEN}',
                'Content-Type': 'application/json',
            },
            json=dict(
                warehouse_id=config.get_warehouse_id(),
                **kwargs,
            )
        )

        response.raise_for_status()
        return response.json()

    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}", e)

    return None

def submit_feedback(text: str, rating: int, comments: str) -> bool:
    """Submit user feedback to Databricks SQL table"""
    statement = """
        INSERT INTO user_feedback (text, rating, comments, created_at) VALUES (:text, :rating, :comments, :created_at)
    """

    parameter_values = [
        {"name": "text", "value": text, "type": "STRING"},
        {"name": "rating", "value": str(rating), "type": "INT"},
        {"name": "comments", "value": comments, "type": "STRING"},
        {"name": "created_at", "value": datetime.now().isoformat(), "type": "TIMESTAMP"}
    ]

    if run_sql(statement=statement, parameters=parameter_values):
        logger.info('Sent user feedback successfully!')
        return True

    return False
