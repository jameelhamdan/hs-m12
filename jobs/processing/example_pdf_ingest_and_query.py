"""
PDF Assistant - Databricks Delta Tables Data Operations
This script demonstrates how to write to and read from the normalized table structure

Table Schema:

-- 1. Main Documents Table
CREATE TABLE documents (
    document_id INT,
    upload_timestamp TIMESTAMP,
    source_language STRING,
    page_count INT,
    word_count INT,
    processing_time_sec DOUBLE,
    created_at TIMESTAMP,
    expires_at TIMESTAMP, -- for 24h auto-deletion
    PRIMARY KEY (document_id)
) USING DELTA;

-- 2. Document Outputs Table  
CREATE TABLE document_outputs (
    document_id INT,
    output_type STRING,  -- 'summary', 'translation'
    target_language STRING,
    output_text STRING,
    word_count INT,
    created_at TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
) USING DELTA;

-- 3. User Feedback Table
CREATE TABLE user_feedback (
    feedback_id INT,
    document_id INT,
    feedback_type STRING, -- 'summary_rating', 'translation_issue', 'tts_rating'
    rating INT,
    comments STRING,
    created_at TIMESTAMP,
    PRIMARY KEY (feedback_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
) USING DELTA;

-- 4. Processing Metrics Table
CREATE TABLE processing_metrics (
    document_id INT,
    metric_type STRING, -- 'rouge_l', 'bleu', 'mos', 'latency'
    metric_value DOUBLE,
    target_language STRING,
    created_at TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
) USING DELTA;
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import datetime, timedelta
import json

# Initialize Spark session (automatically available in Databricks)
# spark = SparkSession.builder.appName("PDFAssistant").getOrCreate()

class PDFAssistantDataLayer:
    
    def __init__(self, catalog="workspace_team1", schema="default"):
        self.catalog = catalog
        self.schema = schema
        self.base_path = f"{catalog}.{schema}"
    
    def _get_next_document_id(self):
        """
        Get next available document ID (auto-increment)
        Thread-safe approach using Delta Lake's ACID properties
        """
        try:
            # Get current max ID
            result = spark.sql(f"""
                SELECT COALESCE(MAX(document_id), 0) + 1 as next_id 
                FROM {self.base_path}.documents
            """).collect()
            return result[0].next_id
        except Exception:
            # If table doesn't exist or is empty, start from 1
            return 1
    
    def _get_next_feedback_id(self):
        """
        Get next available feedback ID (auto-increment)
        """
        try:
            result = spark.sql(f"""
                SELECT COALESCE(MAX(feedback_id), 0) + 1 as next_id 
                FROM {self.base_path}.user_feedback
            """).collect()
            return result[0].next_id
        except Exception:
            return 1
    
    # ============================================================================
    # WRITING DATA - Multiple approaches for backend developers
    # ============================================================================
    
    def insert_document_simple(self, document_data):
        """
        Simple approach: Insert document data one table at a time
        Backend developers will find this familiar and straightforward
        """
        # Generate next document ID (auto-increment)
        doc_id = self._get_next_document_id()
        
        # 1. Insert main document record
        document_row = [
            (doc_id,
             datetime.fromisoformat(document_data['upload_timestamp']),
             document_data['source_language'],
             document_data['processing_metadata']['page_count'],
             document_data['processing_metadata']['word_count'],
             document_data['processing_metadata']['processing_time_sec'],
             datetime.now(),
             datetime.now() + timedelta(hours=24))  # expires in 24h
        ]
        
        doc_df = spark.createDataFrame(document_row, [
            "document_id", "upload_timestamp", "source_language", 
            "page_count", "word_count", "processing_time_sec", 
            "created_at", "expires_at"
        ])
        
        doc_df.write.mode("append").saveAsTable(f"{self.base_path}.documents")
        
        # 2. Insert outputs (summary + translations)
        output_rows = []
        
        # Add summary
        if 'summary' in document_data.get('outputs', {}):
            output_rows.append((
                doc_id, 'summary', document_data['source_language'],
                document_data['outputs']['summary']['text'],
                document_data['outputs']['summary']['length'],
                datetime.now()
            ))
        
        # Add translations
        for translation in document_data.get('outputs', {}).get('translations', []):
            output_rows.append((
                doc_id, 'translation', translation['language'],
                translation['text'], len(translation['text']),
                datetime.now()
            ))
        
        if output_rows:
            outputs_df = spark.createDataFrame(output_rows, [
                "document_id", "output_type", "target_language", 
                "output_text", "word_count", "created_at"
            ])
            outputs_df.write.mode("append").saveAsTable(f"{self.base_path}.document_outputs")
        
        return doc_id
    
    def insert_document_batch(self, documents_list):
        """
        Batch approach: More efficient for multiple documents
        """
        # Prepare all document records
        doc_rows = []
        output_rows = []
        
        for doc_data in documents_list:
            doc_id = self._get_next_document_id()
            
            # Main document
            doc_rows.append((
                doc_id,
                datetime.fromisoformat(doc_data['upload_timestamp']),
                doc_data['source_language'],
                doc_data['processing_metadata']['page_count'],
                doc_data['processing_metadata']['word_count'],
                doc_data['processing_metadata']['processing_time_sec'],
                datetime.now(),
                datetime.now() + timedelta(hours=24)
            ))
            
            # Outputs
            if 'summary' in doc_data.get('outputs', {}):
                output_rows.append((
                    doc_id, 'summary', doc_data['source_language'],
                    doc_data['outputs']['summary']['text'],
                    doc_data['outputs']['summary']['length'],
                    datetime.now()
                ))
        
        # Batch insert
        if doc_rows:
            doc_df = spark.createDataFrame(doc_rows, [
                "document_id", "upload_timestamp", "source_language", 
                "page_count", "word_count", "processing_time_sec", 
                "created_at", "expires_at"
            ])
            doc_df.write.mode("append").saveAsTable(f"{self.base_path}.documents")
        
        if output_rows:
            outputs_df = spark.createDataFrame(output_rows, [
                "document_id", "output_type", "target_language", 
                "output_text", "word_count", "created_at"
            ])
            outputs_df.write.mode("append").saveAsTable(f"{self.base_path}.document_outputs")
    
    def insert_feedback(self, document_id, feedback_data):
        """
        Insert user feedback - simple and straightforward
        """
        feedback_rows = []
        
        if 'summary_rating' in feedback_data:
            feedback_rows.append((
                self._get_next_feedback_id(), document_id, 'summary_rating',
                feedback_data['summary_rating'], None, datetime.now()
            ))
        
        for issue in feedback_data.get('translation_issues', []):
            feedback_rows.append((
                self._get_next_feedback_id(), document_id, 'translation_issue',
                None, issue, datetime.now()
            ))
        
        if feedback_rows:
            feedback_df = spark.createDataFrame(feedback_rows, [
                "feedback_id", "document_id", "feedback_type", 
                "rating", "comments", "created_at"
            ])
            feedback_df.write.mode("append").saveAsTable(f"{self.base_path}.user_feedback")
    
    # ============================================================================
    # READING DATA - Easy queries that backend developers will love
    # ============================================================================
    
    def get_document_complete(self, document_id):
        """
        Get complete document data - returns JSON-like structure
        Backend developers get the familiar nested structure they expect
        """
        # Main document info
        doc_query = f"""
        SELECT * FROM {self.base_path}.documents 
        WHERE document_id = '{document_id}'
        """
        doc_row = spark.sql(doc_query).collect()[0]
        
        # Get all outputs
        outputs_query = f"""
        SELECT output_type, target_language, output_text, word_count
        FROM {self.base_path}.document_outputs 
        WHERE document_id = '{document_id}'
        """
        outputs = spark.sql(outputs_query).collect()
        
        # Get feedback
        feedback_query = f"""
        SELECT feedback_type, rating, comments
        FROM {self.base_path}.user_feedback 
        WHERE document_id = '{document_id}'
        """
        feedback = spark.sql(feedback_query).collect()
        
        # Reconstruct familiar JSON structure
        result = {
            "document_id": doc_row.document_id,
            "upload_timestamp": doc_row.upload_timestamp.isoformat(),
            "source_language": doc_row.source_language,
            "processing_metadata": {
                "page_count": doc_row.page_count,
                "word_count": doc_row.word_count,
                "processing_time_sec": doc_row.processing_time_sec
            },
            "outputs": {
                "summary": None,
                "translations": []
            },
            "feedback": {}
        }
        
        # Fill outputs
        for output in outputs:
            if output.output_type == 'summary':
                result["outputs"]["summary"] = {
                    "text": output.output_text,
                    "length": output.word_count
                }
            elif output.output_type == 'translation':
                result["outputs"]["translations"].append({
                    "language": output.target_language,
                    "text": output.output_text
                })
        
        # Fill feedback
        for fb in feedback:
            if fb.feedback_type == 'summary_rating':
                result["feedback"]["summary_rating"] = fb.rating
            # Add other feedback types as needed
        
        return result
    
    def get_recent_documents(self, limit=10):
        """
        Get recent documents - simple query
        """
        query = f"""
        SELECT document_id, upload_timestamp, source_language, page_count
        FROM {self.base_path}.documents 
        ORDER BY upload_timestamp DESC 
        LIMIT {limit}
        """
        return spark.sql(query).toPandas().to_dict('records')
    
    def get_processing_stats(self, date_from=None):
        """
        Analytics query - easy to write and understand
        """
        date_filter = f"WHERE upload_timestamp >= '{date_from}'" if date_from else ""
        
        query = f"""
        SELECT 
            source_language,
            COUNT(*) as document_count,
            AVG(processing_time_sec) as avg_processing_time,
            AVG(word_count) as avg_word_count
        FROM {self.base_path}.documents 
        {date_filter}
        GROUP BY source_language
        """
        return spark.sql(query).toPandas().to_dict('records')
    
    def search_documents_by_content(self, search_term):
        """
        Search in document outputs - simple text search
        """
        query = f"""
        SELECT DISTINCT d.document_id, d.upload_timestamp, d.source_language
        FROM {self.base_path}.documents d
        JOIN {self.base_path}.document_outputs o ON d.document_id = o.document_id
        WHERE o.output_text LIKE '%{search_term}%'
        ORDER BY d.upload_timestamp DESC
        """
        return spark.sql(query).toPandas().to_dict('records')


# ============================================================================
# CONVENIENCE WRAPPER CLASS - Makes it even easier for backend developers
# ============================================================================

class PDFAssistantAPI:
    """
    High-level API that abstracts away the complexity
    Backend developers can use this without knowing about tables
    """
    
    def __init__(self):
        self.data_layer = PDFAssistantDataLayer()
    
    def save_document(self, document_json):
        """
        Save document - accepts the original JSON structure
        """
        return self.data_layer.insert_document_simple(document_json)
    
    def get_document(self, document_id):
        """
        Get document - returns the original JSON structure
        """
        return self.data_layer.get_document_complete(document_id)
    
    def add_feedback(self, document_id, feedback):
        """
        Add feedback - simple method call
        """
        return self.data_layer.insert_feedback(document_id, feedback)
    
    def get_stats(self):
        """
        Get processing statistics
        """
        return self.data_layer.get_processing_stats()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    Examples showing how backend developers would use this
    """
    
    # Initialize API
    pdf_api = PDFAssistantAPI()
    
    # Example document data (updated with INT IDs)
    document_data = {
        "upload_timestamp": "2024-01-15T10:30:00",
        "source_language": "en",
        "target_languages": ["es", "fr", "ar"],
        "processing_metadata": {
            "page_count": 12,
            "word_count": 4500,
            "processing_time_sec": 8.2
        },
        "outputs": {
            "summary": {
                "text": "This document discusses...",
                "length": 123
            },
            "translations": [
                {
                    "language": "es",
                    "text": "Este documento discute..."
                }
            ]
        },
        "feedback": {
            "summary_rating": 4,
            "translation_issues": []
        }
    }
    
    # Save document - exactly like saving to JSON
    doc_id = pdf_api.save_document(document_data)
    print(f"Saved document: {doc_id}")
    
    # Get document - returns same JSON structure
    retrieved_doc = pdf_api.get_document(doc_id)
    print(f"Retrieved: {retrieved_doc['source_language']}")
    
    # Add feedback - simple
    feedback = {"summary_rating": 5}
    pdf_api.add_feedback(doc_id, feedback)
    
    # Get stats - easy analytics
    stats = pdf_api.get_stats()
    print(f"Processing stats: {stats}")


if __name__ == "__main__":
    example_usage()
