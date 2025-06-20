import streamlit as st
import requests
import uuid
import config
import time
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

current_documents: list[str] = []


def extract_text_from_pdf(file) -> str:
    """Extract text from PDF with error handling."""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
        return full_text
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


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


def upload_section():
    st.header('Upload a File')
    uploaded_files = st.file_uploader('Choose a file', type=['pdf'], accept_multiple_files=True)

    if st.button('Submit', disabled=len(uploaded_files) == 0):
        if len(uploaded_files) > 0:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_files = len(uploaded_files)
                global current_documents
                current_documents = []  # Reset current documents

                for i, file in enumerate(uploaded_files):
                    progress = int((i + 1) / total_files * 100)
                    progress_bar.progress(progress)
                    status_text.text(f'Uploading file {i + 1} of {total_files}: {file.name}')

                    upload(file)
                    current_documents.append(extract_text_from_pdf(file))

                progress_bar.progress(100)
                status_text.text('Upload complete!')
                time.sleep(1)

                st.success('Files Uploaded Successfully!')
                # Reset chat when new documents are uploaded
                if "messages" in st.session_state:
                    del st.session_state.messages

                time.sleep(2)
                progress_bar.empty()
                status_text.empty()

            except Exception as e:
                logger.error(e)
                st.error(f'Error reading file: {str(e)}')
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()


def query_databricks_model(prompt, chat_history):
    """Function to query your Databricks model"""
    try:
        # Prepare the messages in the format your Databricks model expects
        messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]

        # Add system message with current documents if they exist
        if current_documents:
            documents_context = "\n\n".join(current_documents)
            system_message = {
                "role": "system",
                "content": f"""You are a helpful assistant. Here are some documents that have been uploaded for context:
                {documents_context}

                Please use this information when answering questions. If the question is unrelated to these documents, 
                answer based on your general knowledge."""
            }
            messages.insert(0, system_message)

        # Add the new user message
        messages.append({"role": "user", "content": prompt})

        # Make request to Databricks serving endpoint
        response = requests.post(
            f"{config.DATABRICKS_URL}/serving-endpoints/{config.DATABRICKS_MODEL_ENDPOINT}/invocations",
            headers={
                'Authorization': f'Bearer {config.DATABRICKS_API_TOKEN}',
                'Content-Type': 'application/json',
            },
            json={
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7
            }
        )
        response.raise_for_status()

        # Parse the response - adjust this based on your model's response format
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        logger.error(f"Error querying Databricks model: {str(e)}")
        return None


def chat_section():
    st.header("Chat with AI")

    # Add button to reset chat and documents
    if st.button("Reset Chat & Documents"):
        global current_documents
        current_documents = []
        if "messages" in st.session_state:
            del st.session_state.messages
        st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial system message if documents exist
        if current_documents:
            documents_context = "\n\n".join(current_documents)
            st.session_state.messages.append({
                "role": "system",
                "content": f"""You are a helpful assistant. Here are some documents that have been uploaded for context:
                {documents_context}

                Please use this information when answering questions. If the question is unrelated to these documents, 
                answer based on your general knowledge."""
            })

    if "max_messages" not in st.session_state:
        # Counting both user and assistant messages, so 10 rounds of conversation
        st.session_state.max_messages = 20

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't display system messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Check if message limit has been reached
    if len(st.session_state.messages) >= st.session_state.max_messages:
        st.info(
            """Notice: The maximum message limit for this demo version has been reached. We value your interest!
            We encourage you to experience further interactions by building your own application with instructions
            from Streamlit's [Build a basic LLM chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
            tutorial. Thank you for your understanding."""
        )
    else:
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                try:
                    # Get response from Databricks model
                    full_response = query_databricks_model(prompt, st.session_state.messages)

                    if full_response is None:
                        raise Exception("Failed to get response from model")

                    # Display the response
                    st.markdown(full_response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.session_state.max_messages = len(st.session_state.messages)
                    error_message = """
                        Oops! Sorry, I can't talk now. There was an error processing your request.
                        Please try again later.
                    """
                    st.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    logger.error(f"Chat error: {str(e)}")


if __name__ == '__main__':
    upload_section()
    chat_section()
