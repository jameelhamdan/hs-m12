import logging
from functools import lru_cache
import config
import mlflow
import mlflow.genai


logger = logging.getLogger(__name__)

# Prompt templates (will be loaded from MLflow)
PROMPT_DEFAULT_TEMPLATES = {
    "retrieve_document": """Please retrieve document related to following: {text}""",
    "summarize_text": """Please summarize the following text. 
    Keep the summary concise (3-5 sentences) while preserving key information.
    
    Text: {text}""",
    "translate_text": """Translate the following text to "{target_language}" or if not found, by language mentioned by user. 
Maintain the original meaning and tone as closely as possible.

Text: {text}""",
    "agent_system_prompt": """You are a helpful AI assistant that can summarize and translate text. 
You have access to tools that can summarize text, translate text, or do both in sequence.
Use the tools when appropriate to answer user requests.

When you need to use a tool, you will automatically call the appropriate tool with the required parameters.
When you have the final answer, respond normally with the answer.

Tools available:
- RetrieveDocument: For retrieving document based on text search
- Summarize: For summarizing text
- Translate: For translating text to different languages
- SummarizeAndTranslate: For first summarizing and then translating text"""
}

@lru_cache(maxsize=None)
def get_prompt(name: str) -> str:
    prompt_name = f'prompts:/{config.DATABRICKS_CATALOG}.{config.DATABRICKS_SCHEMA}.{name}@latest'
    prompt = mlflow.genai.load_prompt(prompt_name)
    return prompt.template
