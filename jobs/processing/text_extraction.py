import re
import uuid
import os
import shutil
from datetime import datetime
import fitz  # PyMuPDF
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType
from pyspark.sql import functions as F
import logging

logger = logging.getLogger(__name__)

VOLUME_PATH = "/Volumes/workspace_team1/default/documents/"
INPUT_PATH = os.path.join(VOLUME_PATH, 'landing')
OUTPUT_PATH = os.path.join(VOLUME_PATH, 'processed')
ERROR_PATH = os.path.join(VOLUME_PATH, 'errors')
CHECKPOINT_LOCATION = f"{VOLUME_PATH}/_checkpoints"

def setup_tables():
    spark.sql("""
    CREATE TABLE IF NOT EXISTS document (
        id long PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        document_uuid STRING,
        upload_timestamp TIMESTAMP,
        page_count INT,
        word_count INT,
        created_at TIMESTAMP,
        expires_at TIMESTAMP,
        text STRING,
        file_name STRING,
        file_path STRING
    )
    USING DELTA
    """)

    spark.sql("""
    CREATE TABLE IF NOT EXISTS document_processing_error (
        id long PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        error_timestamp TIMESTAMP,
        file_name STRING,
        file_path STRING,
        error_message STRING,
        stack_trace STRING
    )
    USING DELTA
    """)


def ensure_directory_exists(path):
    """Ensure the directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def move_file(src_path, dest_dir):
    """Move file to destination directory, preserving filename."""
    ensure_directory_exists(dest_dir)
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)

    # Handle case where file already exists in destination
    counter = 1
    while os.path.exists(dest_path):
        name, ext = os.path.splitext(filename)
        dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
        counter += 1

    shutil.move(src_path, dest_path)
    return dest_path


def extract_document_uuid(file_path):
    """Extract UUID from filename or generate a new one."""
    UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
    filename = os.path.basename(file_path)
    match = UUID_PATTERN.search(filename)
    return match.group(0) if match else str(uuid.uuid4())


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with error handling."""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_document = fitz.open(stream=f.read(), filetype="pdf")
            full_text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                full_text += page.get_text()
            return full_text
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def get_page_count(pdf_path):
    """Get page count from PDF."""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_document = fitz.open(stream=f.read(), filetype="pdf")
            return len(pdf_document)
    except Exception as e:
        logger.error(e)
        raise Exception(f"Failed to get page count: {str(e)}")


def get_expiry_date():
    """Calculate expiry date (24 hours from now)"""
    from datetime import timedelta
    return datetime.now() + timedelta(hours=24)


def process_pdf_file(file_path):
    """Process a single PDF file and save to Delta table with UUID."""
    start_time = datetime.now()
    current_time = datetime.now()
    document_uuid = extract_document_uuid(file_path)
    file_name = os.path.basename(file_path)

    try:
        # Extract text and metadata
        raw_text = extract_text_from_pdf(file_path)
        word_count = len(raw_text.split())
        page_count = get_page_count(file_path)
        expiry_date = get_expiry_date()
        processing_time_sec = int((datetime.now() - start_time).total_seconds())

        # Define the schema explicitly
        schema = StructType([
            StructField("document_uuid", StringType(), False),
            StructField("upload_timestamp", TimestampType(), False),
            StructField("page_count", IntegerType(), False),
            StructField("word_count", IntegerType(), False),
            StructField("created_at", TimestampType(), False),
            StructField("expires_at", TimestampType(), False),
            StructField("text", StringType(), False),
            StructField("file_name", StringType(), False),
            StructField("file_path", StringType(), False)
        ])

        # Create a row with the data
        row = (
            document_uuid,
            current_time,
            page_count,
            word_count,
            current_time,
            expiry_date,
            raw_text,
            file_name,
            file_path
        )

        # Create DataFrame with explicit schema
        df = spark.createDataFrame([row], schema=schema)

        # Write to Delta table
        df.write.format("delta").mode("append").saveAsTable("document")

        # Move to processed directory after successful processing
        processed_path = move_file(file_path, OUTPUT_PATH)
        print(f"Successfully processed {file_name} (document_id: {document_uuid}, uuid: {document_uuid})")
        print(f"Moved file to: {processed_path}")

    except Exception as e:
        logger.error(e)
        # Move to error directory if processing fails
        error_path = '' # move_file(file_path, ERROR_PATH)
        error_msg = str(e)
        stack_trace = traceback.format_exc()

        # Log error to error table
        error_row = (
            str(uuid.uuid4()),
            current_time,
            file_name,
            file_path,
            error_msg,
            stack_trace
        )

        error_schema = StructType([
            StructField("error_timestamp", TimestampType(), False),
            StructField("file_name", StringType(), False),
            StructField("file_path", StringType(), False),
            StructField("error_message", StringType(), False),
            StructField("stack_trace", StringType(), False)
        ])

        error_df = spark.createDataFrame([error_row], schema=error_schema)
        error_df.write.format("delta").mode("append").saveAsTable("document_processing_errors")

        print(f"Failed to process {file_name}: {error_msg}")
        print(f"Moved failed file to: {error_path}")


def run():
    """Find and process new PDF files in the Volume."""
    # Get all PDF files in the Volume
    ensure_directory_exists(OUTPUT_PATH)
    ensure_directory_exists(ERROR_PATH)

    new_files = []
    for root, dirs, files in os.walk(INPUT_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                new_files += [os.path.join(root, file)]

    if not new_files:
        print("No new PDF files to process")
        return

    print(f"Found {len(new_files)} new PDF files to process")

    # Process each file
    for file_path in new_files:
        try:
            process_pdf_file(file_path)
        except Exception as e:
            # Error already logged in process_pdf_file
            continue


run()