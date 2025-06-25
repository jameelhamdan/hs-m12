import re
import config
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from databricks_langchain.chat_models import ChatDatabricks
from prompts import get_prompt
import logging


logger = logging.getLogger(__name__)

__all__ = ['call_agent']

# Chat models configuration
chat_base_model = ChatDatabricks(
    endpoint=config.DATABRICKS_MODEL_BASE_ENDPOINT,
    api_token=config.DATABRICKS_TOKEN,
    max_tokens=1024,
    temperature=0.1
)

chat_rag_model = ChatDatabricks(
    endpoint=config.DATABRICKS_MODEL_RAG_ENDPOINT,
    api_token=config.DATABRICKS_TOKEN,
    max_tokens=1024,
    temperature=0.1
)

def retrieve_document(text: str) -> str:
    """Summarize text in the specified language"""
    try:
        prompt = get_prompt('retrieve_document').format(text=text)
        response = chat_rag_model.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return f"Error generating summary: {str(e)}"


def summarize_text(text: str) -> str:
    """Summarize text in the specified language"""
    try:
        prompt = get_prompt('summarize_text').format(text=text)
        response = chat_base_model.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return f"Error generating summary: {str(e)}"


def translate_text(text: str, target_language: str = None) -> str:
    """Translate text to the target language"""
    try:
        prompt = get_prompt('translate_text').format(
            text=text,
            target_language=target_language
        )
        response = chat_base_model.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"Error translating text: {str(e)}"


def summarize_and_translate(text: str, target_language: str = None) -> str:
    """Summarize and then translate the text"""
    try:
        summary = summarize_text(text)
        return translate_text(summary, target_language)
    except Exception as e:
        logger.error(f"Error in summarize_and_translate: {str(e)}")
        return f"Error in summarize_and_translate: {str(e)}"


# Define tools with proper parameter schemas
tools = [
    Tool(
        name="RetrieveDocument",
        func=retrieve_document,
        description='Useful for searching and retrieving documents by text search',
        args_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to search for documents with"}
            },
            "required": ["text"]
        }
    ),
    Tool(
        name="Summarize",
        func=summarize_text,
        description='Useful for summarizing text in various languages.',
        args_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to summarize"}
            },
            "required": ["text"]
        }
    ),
    Tool(
        name="Translate",
        func=translate_text,
        description='Useful for translating text to different languages.',
        args_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate"},
                "target_language": {"type": "string", "description": "Target language code"}
            },
            "required": ["text", "target_language"]
        }
    ),
    Tool(
        name="SummarizeAndTranslate",
        func=summarize_and_translate,
        description='Useful for first summarizing and then translating text.',
        args_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to process"},
                "target_language": {"type": "string", "description": "Target language code"}
            },
            "required": ["text", "target_language"]
        }
    ),
]

# Initialize the agent (will be done after prompts are loaded)
agent_executor = None


def initialize_agent():
    """Initialize the agent after prompts are loaded"""
    global agent_executor
    prompt = get_prompt("agent_system_prompt")

    # Define the prompt template for the tool calling agent
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # Create the tool calling agent
    agent = create_tool_calling_agent(
        llm=chat_rag_model,
        tools=tools,
        prompt=agent_prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


def call_agent(prompt: str, current_documents: list[str] = None, chat_history: list[tuple[str, str]] = None) -> str:
    """
    Execute the agent with the given prompt, documents, and chat history

    Args:
        prompt: The user's input prompt
        current_documents: List of document contents (optional)
        chat_history: List of tuples representing (human_message, ai_message) history

    Returns:
        The agent's response
    """
    try:
        if agent_executor is None:
            initialize_agent()

        # Format chat history for LangChain
        formatted_history = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                formatted_history.extend([
                    HumanMessage(content=human_msg),
                    AIMessage(content=ai_msg)
                ])

        input_with_context = prompt
        if current_documents:
            docs_context = "\n\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(current_documents)])
            input_with_context = f"Context documents:\n{docs_context}\n\nUser question: {prompt}"

        response = agent_executor.invoke({
            "input": input_with_context,
            "chat_history": formatted_history,
        })
        output = response.get("output", "No response generated")
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        output = re.sub(r'Thought:.*?Observation:', '', output, flags=re.DOTALL)
        output = re.sub(r'Action:.*?Action Input:.*?Observation:', '', output, flags=re.DOTALL)

        return output.strip()

    except Exception as e:
        logger.error(f"Error in call_agent: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"
