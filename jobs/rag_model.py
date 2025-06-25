from databricks.vector_search.client import VectorSearchClient
from databricks_langchain.embeddings import DatabricksEmbeddings
from databricks_langchain.chat_models import ChatDatabricks
from mlflow.pyfunc import PythonModel
from typing import Dict, Any, Iterator, List, Union, Optional
import cloudpickle
import mlflow
from mlflow.utils.environment import _mlflow_conda_env
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mlflow.pyfunc import PythonModelContext
import pandas as pd
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessageChunk

# Enable autologging
mlflow.langchain.autolog()


class RAGChatbotModel(PythonModel):
    def __init__(self, catalog, db, vector_search_endpoint_name="pdf-vector-endpoint"):
        self.catalog = catalog
        self.db = db
        self.vector_search_endpoint_name = vector_search_endpoint_name
        self.vector_search_index = f"{catalog}.{db}.document_chunks_vs_index"

        # Initialize models with and without streaming
        self.embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")

        self.vsc = VectorSearchClient()
        self.index = self.vsc.get_index(
            endpoint_name=self.vector_search_endpoint_name,
            index_name=self.vector_search_index
        )

        # Regular LLM for MLflow serving
        self.llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            max_tokens=500,
            temperature=0.1,
            streaming=False
        )

        # Separate LLM instance for streaming (used outside MLflow)
        self.streaming_llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            max_tokens=500,
            temperature=0.1,
            streaming=True
        )

        # Prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided context.

            Context: {context}

            Question: {question}

            Answer: """
        )

        # Regular chain for MLflow
        self.rag_chain = (
                RunnableLambda(self.format_input)
                | self.prompt_template
                | self.llm
                | StrOutputParser()
        )

        # Streaming chain (not used in MLflow serving)
        self.streaming_chain = (
                RunnableLambda(self.format_input)
                | self.prompt_template
                | self.streaming_llm
        )

    def format_input(self, input_dict: Dict[str, str]) -> Dict[str, str]:
        """Helper method to format input for the chain"""
        query = input_dict["question"]
        context = self.retrieve_context(query)
        return {"context": context, "question": query}

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context using vector search"""
        query_vector = self.embedding_model.embed_query(query)
        results = self.index.similarity_search(
            query_vector=query_vector,
            columns=["chunk_text", "document_id"],
            num_results=k
        )
        if results and 'result' in results and 'data_array' in results['result']:
            return "\n\n".join([row[0] for row in results['result']['data_array']])
        return ""

    def generate_response(self, query: str) -> str:
        """Generate response using LLM with retrieved context"""
        return self.rag_chain.invoke({"question": query})

    def stream_response(self, query: str) -> Iterator[str]:
        """Stream response tokens as they're generated (for direct Python usage)"""
        for chunk in self.streaming_chain.stream({"question": query}):
            if isinstance(chunk, AIMessageChunk):
                yield chunk.content
            else:
                yield str(chunk)

    def predict(
            self,
            context: PythonModelContext,
            model_input: pd.DataFrame
    ) -> List[str]:
        """
        Generate responses for input queries.
        MLflow serving doesn't support streaming, so we only implement batch predict.

        Args:
            context: MLflow context object
            model_input: DataFrame with queries in first column

        Returns:
            List of generated responses
        """
        queries = model_input.iloc[:, 0].tolist()
        return [self.generate_response(query) for query in queries]

    def predict_stream(
            self,
            model_input: pd.DataFrame,
            **kwargs: Any
    ) -> Iterator[List[str]]:
        """
        Generate streaming responses for input queries.
        Yields responses token by token for each query in the input.

        Args:
            model_input: DataFrame with queries in first column
            kwargs: Additional arguments (ignored in this implementation)

        Yields:
            Iterator of lists containing the latest tokens for each query
        """
        queries = model_input.iloc[:, 0].tolist()

        # Create a list of generators, one for each query
        streams = [self.stream_response(query) for query in queries]

        # Initialize response buffers for each query
        responses = ["" for _ in queries]

        # Continue until all streams are exhausted
        while True:
            any_active = False
            current_tokens = ["" for _ in queries]

            # Get next token from each active stream
            for i, stream in enumerate(streams):
                try:
                    token = next(stream)
                    current_tokens[i] = token
                    responses[i] += token
                    any_active = True
                except StopIteration:
                    pass

            if not any_active:
                break

            yield current_tokens


def register_and_deploy_model():
    catalog = "workspace_team1"
    db = "default"
    vector_search_endpoint_name = "pdf-vector-endpoint"
    model_name = f"{catalog}.{db}.rag_chatbot_model"
    endpoint_name = "pdf-chatbot-endpoint"

    rag_model = RAGChatbotModel(catalog, db, vector_search_endpoint_name)

    # Create model with explicit type handling
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            streamable=True,
            python_model=rag_model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=ModelSignature(
                inputs=Schema([ColSpec("string", "query")]),
                outputs=Schema([ColSpec("string")])
            ),
            input_example=pd.DataFrame({"query": ["What is RAG?"]}),
            resources=[
                DatabricksVectorSearchIndex(index_name=f"{catalog}.{db}.document_chunks_vs_index"),
                DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct")
            ]
        )

    # Get registered model version
    registered_model = mlflow.register_model(model_info.model_uri, model_name)

    # Deploy with proper ServedModelInput
    w = WorkspaceClient()
    served_models = [
        ServedModelInput(
            model_name=registered_model.name,
            model_version=registered_model.version,
            workload_size="Small",
            scale_to_zero_enabled=False
        )
    ]

    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_models=served_models
    )

    print(f"Model registered as {model_name} and deployed to endpoint {endpoint_name}")
    return model_info

