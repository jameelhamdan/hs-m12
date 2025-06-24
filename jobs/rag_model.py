from databricks.vector_search.client import VectorSearchClient
from databricks_langchain.embeddings import DatabricksEmbeddings
from databricks_langchain.chat_models import ChatDatabricks
from mlflow.pyfunc import PythonModel
from typing import Dict, Any
import mlflow
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RAGChatbot:
    def __init__(self, catalog, db, vector_search_endpoint_name="pdf-vector-endpoint"):
        self.catalog = catalog
        self.db = db
        self.vector_search_endpoint_name = vector_search_endpoint_name
        self.vector_search_index = f"{catalog}.{db}.document_chunks_vs_index"

        # Initialize embedding model
        self.embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")

        # Initialize vector search client
        self.vsc = VectorSearchClient()
        self.index = self.vsc.get_index(
            endpoint_name=self.vector_search_endpoint_name,
            index_name=self.vector_search_index
        )

        self.llm = ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            max_tokens=500,
            temperature=0.1
        )

        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided context.

            Context: {context}

            Question: {question}

            Answer: """
        )
        self.rag_chain = (
                self.prompt_template
                | self.llm
                | StrOutputParser()
        )

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context using vector search"""
        query_vector = self.embedding_model.embed_query(query)
        results = self.index.similarity_search(
            query_vector=query_vector,
            columns=["chunk_text", "document_id"],
            num_results=k
        )
        # Properly handle the vector search results format
        if results and 'result' in results and 'data_array' in results['result']:
            return "\n\n".join([row[0] for row in results['result']['data_array']])
        return ""

    def generate_response(self, query: str) -> str:
        """Generate response using LLM with retrieved context"""
        context = self.retrieve_context(query)
        return self.rag_chain.invoke({"context": context, "question": query})


class RAGChatbotModel(PythonModel):
    def __init__(self, catalog, db, vector_search_endpoint_name):
        self.rag_chatbot = RAGChatbot(catalog, db, vector_search_endpoint_name)

    def predict(self, context, model_input):
        if isinstance(model_input, str):
            queries = [model_input]
        elif isinstance(model_input, dict):
            queries = [model_input["query"]]
        else:
            queries = model_input.iloc[:, 0].tolist()

        return [self.rag_chatbot.generate_response(query) for query in queries]


# Register and deploy the model
def register_and_deploy_model():
    # Set parameters
    catalog = "workspace_team1"
    db = "default"
    vector_search_endpoint_name = "pdf-vector-endpoint"
    model_name = "workspace_team1.default.rag_chatbot_model"

    # Create the model
    rag_model = RAGChatbotModel(catalog, db, vector_search_endpoint_name)

    signature = mlflow.models.infer_signature(
        {"query": "What is RAG?"},
        {"response": "Sample response from your model"}
    )

    # Log the model to MLflow
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name='rag_chatbot_model',
            python_model=rag_model,
            registered_model_name=model_name,
            signature=signature,
            input_example={"query": "What is RAG?"},
        )

    w = WorkspaceClient()

    # Check if endpoint exists
    endpoint_name = "pdf-chatbot-endpoint"
    served_models = [
        ServedModelInput(
            model_name=model_name,
            # model_version="latest",
            model_version=1,
            workload_size="Small",
            scale_to_zero_enabled=True
        )
    ]

    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_models=served_models,
    )

    print(f"Model registered as {model_name} and deployed to endpoint {endpoint_name}")
    return model_info

# Call the function to register and deploy
# register_and_deploy_model()


print("\n=== Testing Model Loading ===")
model_info = register_and_deploy_model()
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
test_input = {
    "query": "What is data processing?"
}
print(loaded_model.predict(test_input))
