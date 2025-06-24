import config
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from databricks_langchain.chat_models import ChatDatabricks
import logging


__all__ = ['call_agent']


logger = logging.getLogger(__name__)

chat_base_model = ChatDatabricks(
    endpoint=config.DATABRICKS_MODEL_BASE_ENDPOINT,
    api_token=config.DATABRICKS_API_TOKEN,
    max_tokens=1024,
    temperature=0.1
)

chat_rag_model = ChatDatabricks(
    endpoint=config.DATABRICKS_MODEL_RAG_ENDPOINT,
    api_token=config.DATABRICKS_API_TOKEN,
    max_tokens=1024,
    temperature=0.1
)


def summarize_text(text: str) -> str:
    """Summarize text in the specified language"""
    try:
        prompt = f"""
        Please summarize the following text. 
        Keep the summary concise (3-5 sentences) while preserving key information.

        Text: {text}
        """

        response = chat_rag_model.invoke(prompt)
        # TODO: LOG SUMMARIZE
        # TODO: CALCULATE BLEU OR OTHER METRIC
        return response.content
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return f"Error generating summary: {str(e)}"


def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language"""
    try:
        prompt = f"""
        Translate the following text to {target_language} or if not found, by language mentioned by user. 
        Maintain the original meaning and tone as closely as possible.

        Text: {text}
        """
        response = chat_rag_model.invoke(prompt)
        # TODO: LOG TRANSLATE
        # TODO: CALCULATE BLEU OR OTHER METRIC
        return response.content
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return f"Error translating text: {str(e)}"


def summarize_and_translate(text: str, target_language: str) -> str:
    """Summarize and then translate the text"""
    try:
        summary = summarize_text(text)
        return translate_text(summary, target_language)
    except Exception as e:
        logger.error(f"Error in summarize_and_translate: {str(e)}")
        return f"Error in summarize_and_translate: {str(e)}"


tools = [
    Tool(
        name="Summarize",
        func=summarize_text,
        description='Useful for summarizing text in various languages. Input should be the text to summarize.'
    ),
    Tool(
        name="Translate",
        func=translate_text,
        description='Useful for translating text to different languages. Input should be a dictionary with "text" and "target_language" keys.'
    ),
    Tool(
        name="SummarizeAndTranslate",
        func=summarize_and_translate,
        description='Useful for first summarizing and then translating text. Input should be a dictionary with "text" and "target_language" keys.'
    ),
]

# Define the prompt template for the tool calling agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that can summarize and translate text. 
You have access to tools that can summarize text, translate text, or do both in sequence.
Use the tools when appropriate to answer user requests.

When you need to use a tool, you will automatically call the appropriate tool with the required parameters.
When you have the final answer, respond normally with the answer.

Tools available:
- Summarize: For summarizing text
- Translate: For translating text to different languages
- SummarizeAndTranslate: For first summarizing and then translating text"""),
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
        # TODO: CALCULATE RELEVANCE METRIC
        return response.get("output", "No response generated")

    except Exception as e:
        logger.error(f"Error in call_agent: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"