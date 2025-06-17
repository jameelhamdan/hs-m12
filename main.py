import streamlit as st
import requests
import uuid
import config
import time
import logging

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


def main():
    st.header('Upload a File')
    uploaded_files = st.file_uploader('Choose a file', type=['pdf'], accept_multiple_files=True)

    if st.button('Submit', disabled=len(uploaded_files) == 0):
        if len(uploaded_files) > 0:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_files = len(uploaded_files)

                for i, file in enumerate(uploaded_files):
                    progress = int((i + 1) / total_files * 100)
                    progress_bar.progress(progress)
                    status_text.text(f'Uploading file {i + 1} of {total_files}: {file.name}')

                    upload(file)

                progress_bar.progress(100)
                status_text.text('Upload complete!')
                time.sleep(1)

                st.success('Files Uploaded Successfully!')

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


if __name__ == '__main__':
    main()