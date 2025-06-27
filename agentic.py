import re
import config
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from databricks_langchain.chat_models import ChatDatabricks
from prompts import get_prompt
import logging
from typing import List, Dict, Any, Optional

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


# Define specialized agents
class DocumentRetrievalAgent:
    """Agent specialized in document retrieval tasks"""

    def __init__(self):
        self.tools = [
            Tool(
                name="RetrieveDocument",
                func=self.retrieve_document,
                description='Search and retrieve documents by text search',
            )
        ]
        self.agent = self._create_agent()

    def retrieve_document(self, text: str) -> str:
        """Retrieve documents based on text search"""
        try:
            prompt = get_prompt('retrieve_document').format(text=text)
            response = chat_rag_model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return f"Error retrieving documents: {str(e)}"

    def _create_agent(self):
        prompt = get_prompt("document_retrieval_agent_prompt")
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        return create_tool_calling_agent(
            llm=chat_base_model,
            tools=self.tools,
            prompt=agent_prompt,
        )

    def run(self, input: Dict[str, Any]) -> str:
        executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        response = executor.invoke(input)
        return response.get("output", "No response generated")


class SummarizationAgent:
    """Agent specialized in text summarization tasks"""

    def __init__(self):
        self.tools = [
            Tool(
                name="Summarize",
                func=self.summarize_text,
                description='Summarize text in various languages',
            )
        ]
        self.agent = self._create_agent()

    def summarize_text(self, text: str) -> str:
        """Summarize text"""
        try:
            prompt = get_prompt('summarize_text').format(text=text)
            response = chat_base_model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _create_agent(self):
        prompt = get_prompt("summarization_agent_prompt")
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        return create_tool_calling_agent(
            llm=chat_base_model,
            tools=self.tools,
            prompt=agent_prompt,
        )

    def run(self, input: Dict[str, Any]) -> str:
        executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        response = executor.invoke(input)
        return response.get("output", "No response generated")


class TranslationAgent:
    """Agent specialized in translation tasks"""

    def __init__(self):
        self.tools = [
            Tool(
                name="Translate",
                func=self.translate_text,
                description='Translate text to different languages',
            )
        ]
        self.agent = self._create_agent()

    def translate_text(self, text: str, target_language: str = None) -> str:
        """Translate text to target language"""
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

    def _create_agent(self):
        prompt = get_prompt("translation_agent_prompt")
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        return create_tool_calling_agent(
            llm=chat_base_model,
            tools=self.tools,
            prompt=agent_prompt,
        )

    def run(self, input: Dict[str, Any]) -> str:
        executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        response = executor.invoke(input)
        return response.get("output", "No response generated")


from langchain_core.tools import tool
from typing import Dict, Any


class OrchestratorAgent:
    """Main agent that routes tasks to specialized agents"""

    def __init__(self):
        self.agents = {
            "document": DocumentRetrievalAgent(),
            "summarization": SummarizationAgent(),
            "translation": TranslationAgent()
        }

        # Define tools with proper schemas
        self.tools = [
            self._create_document_tool(),
            self._create_summarization_tool(),
            self._create_translation_tool()
        ]

        self.agent = self._create_orchestrator()

    def _create_document_tool(self):
        @tool
        def document_retrieval(query: str) -> str:
            """Search for and retrieve documents based on a query"""
            return self._call_document_agent(query)

        return document_retrieval

    def _create_summarization_tool(self):
        @tool
        def text_summarization(query: str) -> str:
            """Summarize text or summarize and translate text"""
            return self._call_summarization_agent(query)

        return text_summarization

    def _create_translation_tool(self):
        @tool
        def text_translation(query: str) -> str:
            """Translate text to another language"""
            return self._call_translation_agent(query)

        return text_translation

    def _create_orchestrator(self):
        prompt = get_prompt("orchestrator_agent_prompt")
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        return create_tool_calling_agent(
            llm=chat_base_model,
            tools=self.tools,
            prompt=agent_prompt,
        )

    def _call_document_agent(self, query: str) -> str:
        """Route document-related tasks to document agent"""
        return self.agents["document"].run({
            "input": query,
            "chat_history": []
        })

    def _call_summarization_agent(self, query: str) -> str:
        """Route summarization tasks to summarization agent"""
        return self.agents["summarization"].run({
            "input": query,
            "chat_history": []
        })

    def _call_translation_agent(self, query: str) -> str:
        """Route translation tasks to translation agent"""
        return self.agents["translation"].run({
            "input": query,
            "chat_history": []
        })

    def run(self, input: Dict[str, Any]) -> str:
        executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        response = executor.invoke(input)
        return self._clean_response(response.get("output", "No response generated"))

    def _clean_response(self, output: str) -> str:
        """Clean up agent response"""
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        output = re.sub(r'Thought:.*?Observation:', '', output, flags=re.DOTALL)
        output = re.sub(r'Action:.*?Action Input:.*?Observation:', '', output, flags=re.DOTALL)
        return output.strip()


orchestrator = OrchestratorAgent()


def call_agent(
        prompt: str,
        current_documents: Optional[List[str]] = None,
        chat_history: Optional[List[tuple[str, str]]] = None
) -> str:
    """
    Execute the multi-agent system with the given prompt, documents, and chat history

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

        if current_documents:
            docs_context = "\n\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(current_documents)])
            formatted_history.extend([
                HumanMessage(content=f"Answer questions based on these Context documents:\n{docs_context}\n"),
                AIMessage(content='Context Documents Acknowledged.'),
            ])

        if chat_history:
            for human_msg, ai_msg in chat_history:
                formatted_history.extend([
                    HumanMessage(content=human_msg),
                    AIMessage(content=ai_msg)
                ])

        response = orchestrator.run({
            "input": prompt,
            "chat_history": formatted_history,
        })

        return response

    except Exception as e:
        logger.error(f"Error in call_agent: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"
