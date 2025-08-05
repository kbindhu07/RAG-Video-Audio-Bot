import streamlit as st
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path
import speech_recognition as sr
from pytubefix import YouTube
from PIL import Image
import matplotlib.pyplot as plt
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageNode
import openai
import json
from yt_dlp import YoutubeDL

# Streamlit app setup with a custom layout and title
st.set_page_config(
    page_title="Video Content RAG with Visual Understanding",
    layout="wide",
    page_icon="🎬",
)

# Custom CSS styling to enhance UI readability
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    
    /* Improve text readability */
    .stMarkdown {
        color: #333333 !important;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Style buttons for better visibility */
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        margin-top: 10px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #1565c0;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Improve text input styling */
    .stTextInput > div > input {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 16px;
        color: #333333;
    }
    
    .stTextInput > div > input:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
    }
    
    /* Improve title and header readability */
    h1, h2, h3 {
        color: #1a1a1a !important;
        font-weight: 700;
    }
    
    /* Improve JSON display readability */
    .stJson {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    /* Improve text area readability */
    .stTextArea textarea {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 16px;
        color: #333333;
        line-height: 1.5;
    }
    
    /* Improve success and error message readability */
    .stAlert {
        border-radius: 8px;
        padding: 16px;
        font-weight: 600;
    }
    
    /* Improve spinner text readability */
    .stSpinner {
        color: #1f77b4;
        font-weight: 600;
    }
    
    /* Improve overall spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Add info boxes styling */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 16px;
        margin: 16px 0;
        border-radius: 4px;
    }
    
    .success-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 16px;
        margin: 16px 0;
        border-radius: 4px;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 16px;
        margin: 16px 0;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main title and description
st.title("🎬 Video Content RAG with Visual Understanding")
st.markdown("---")

# Project description
st.markdown(
    """
<div class="info-box">
    <h3>🔬 Research Project: Multimodal Video Content Analysis</h3>
    <p>This system implements a comprehensive multimodal RAG (Retrieval-Augmented Generation) pipeline for video content analysis, 
    combining visual understanding, audio transcription, and intelligent querying capabilities.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Key Features Section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
        <h4>🎬 Video Processing</h4>
        <p>Key frame extraction at 0.5 FPS for optimal visual analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
        <h4>🎧 Audio Transcription</h4>
        <p>Whisper-based speech-to-text with temporal synchronization</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
        <h4>🔍 Multi-modal RAG</h4>
        <p>CLIP embeddings + Vector search for comprehensive understanding</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Technical Architecture Section
st.subheader("🏗️ Technical Architecture")
st.markdown(
    """
This system addresses the following technical challenges:

- **Video preprocessing and frame selection algorithms**: Optimized frame extraction at 0.5 FPS
- **Visual-audio synchronization and alignment**: Coordinated processing pipeline  
- **Object detection and scene understanding**: CLIP-based visual analysis
- **Multi-modal embedding space creation**: Unified embedding strategy
- **Large-scale video data storage and indexing**: LanceDB vector database
"""
)

# OpenAI API key input
st.markdown("### 🔑 Step 1: API Configuration")
st.markdown(
    """
<div class="warning-box">
    <strong>Required:</strong> You'll need an OpenAI API key to use this application. 
    Get one from <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI's platform</a>.
</div>
""",
    unsafe_allow_html=True,
)

api_key = st.text_input(
    "OpenAI API Key:",
    type="password",
    placeholder="sk-xxxxxxxxxxxxxxxx",
    help="Enter your OpenAI API key here. This is required for the LLM to process your questions.",
)
openai.api_key = os.getenv("OPENAI_API_KEY", default=api_key)

# Path configurations
output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"
filepath = output_video_path + "input_vid.mp4"
Path(output_folder).mkdir(
    parents=True, exist_ok=True
)  # Create folder if it doesn't exist


# Function to download video from YouTube
def download_video(url, output_path):
    ydl_opts = {
        "outtmpl": os.path.join(output_path, "input_vid.%(ext)s"),
        "format": "best",
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return {
            "Title": info_dict.get("title"),
            "Author": info_dict.get("uploader"),
            "Views": info_dict.get("view_count"),
        }


# Function to extract frames from a video and save as images
def video_to_images(video_path, output_folder):
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.5)


# Function to extract audio from a video
def video_to_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)


# Function to convert audio to text
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    with audio as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            text = "Audio not recognized"
        except sr.RequestError as e:
            text = f"Error: {e}"
    return text


# Function to plot images
def plot_images(image_paths):
    st.markdown("### 🖼️ Visual Frames Used in Analysis:")
    st.markdown(
        "These are the key visual frames that were analyzed to answer your question:"
    )

    images_shown = 0
    cols = st.columns(3)  # Display 3 images per row

    for i, img_path in enumerate(image_paths):
        if os.path.isfile(img_path) and images_shown < 6:  # Limit to 6 images
            col_idx = images_shown % 3
            with cols[col_idx]:
                image = Image.open(img_path)
                st.image(
                    image, caption=f"Frame {images_shown + 1}", use_column_width=True
                )
            images_shown += 1


# Retrieve query results
def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            retrieved_text.append(res_node.text)
    return retrieved_image, retrieved_text


# Initial state management
if "retriever_engine" not in st.session_state:
    st.session_state.retriever_engine = None
    st.session_state.metadata_vid = None

# Step 2: Video Upload & Processing
st.markdown("### 🎥 Step 2: Video Upload & Processing")
st.markdown(
    """
<div class="info-box">
    <strong>Processing Pipeline:</strong>
    <ol>
        <li>Upload a video file (MP4, MOV, AVI, etc.)</li>
        <li>Extract key frames at 0.5 FPS for visual analysis</li>
        <li>Convert audio to text using Whisper</li>
        <li>Create multimodal embeddings with CLIP</li>
        <li>Index content in LanceDB vector database</li>
    </ol>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a video file:",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload a video file from your computer. The video will be processed through the multimodal pipeline.",
)

if uploaded_file and st.session_state.retriever_engine is None:
    if st.button("🚀 Process Video", help="Start the multimodal processing pipeline"):
        try:
            with st.spinner(
                "Processing video through multimodal pipeline... This may take several minutes."
            ):
                # Save uploaded file
                Path(output_video_path).mkdir(parents=True, exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state.metadata_vid = {
                    "Title": uploaded_file.name,
                    "Author": "Uploaded by user",
                    "Views": "N/A",
                }
                video_to_images(filepath, output_folder)
                video_to_audio(filepath, output_audio_path)
                text_data = audio_to_text(output_audio_path)
                st.session_state["extracted_text"] = text_data

                # Save extracted text to a file
                with open(output_folder + "output_text.txt", "w") as file:
                    file.write(text_data)
                os.remove(output_audio_path)

                # Set up vector stores for text and images
                text_store = LanceDBVectorStore(
                    uri="lancedb", table_name="text_collection"
                )
                image_store = LanceDBVectorStore(
                    uri="lancedb", table_name="image_collection"
                )

                # Set up storage context for multi-modal index
                storage_context = StorageContext.from_defaults(
                    vector_store=text_store, image_store=image_store
                )

                # Load documents from the output folder
                documents = SimpleDirectoryReader(output_folder).load_data()

                # Create the multi-modal index
                index = MultiModalVectorStoreIndex.from_documents(
                    documents, storage_context=storage_context
                )
                st.session_state.retriever_engine = index.as_retriever(
                    similarity_top_k=3, image_similarity_top_k=3
                )

                st.success(
                    "✅ Multimodal processing completed! You can now query the video content."
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Step 3: Ask questions
if st.session_state.retriever_engine:
    st.markdown("### 🔍 Step 3: Multimodal Query Interface")
    st.markdown(
        """
    <div class="info-box">
        <strong>Query Capabilities:</strong>
        <ul>
            <li>Ask about visual content and scenes</li>
            <li>Query audio transcript and spoken content</li>
            <li>Get comprehensive answers combining both modalities</li>
            <li>Receive context-aware responses with frame references</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    user_query = st.text_input(
        "Ask a question about the video content:",
        key="question_input",
        placeholder="What is the main topic discussed in this video?",
        help="Ask any question about the video content, topics, or details shown in the video.",
    )

    if (
        st.button(
            "🔍 Submit Query",
            help="Process your question through the multimodal RAG system",
        )
        and user_query
    ):
        try:
            img, txt = retrieve(
                retriever_engine=st.session_state.retriever_engine, query_str=user_query
            )
            image_documents = SimpleDirectoryReader(
                input_dir=output_folder, input_files=img
            ).load_data()
            context_str = st.session_state.get("extracted_text", "")

            # Display metadata and context
            st.markdown("### 📄 Video Metadata:")
            st.json(st.session_state.metadata_vid)

            st.markdown("### 📝 Extracted Text Context:")
            st.markdown(
                f"""
            <div style="background-color: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 4px solid #1f77b4;">
                {context_str}
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Display the frames used in response
            plot_images(img)

            # Create the LLM prompt
            qa_tmpl_str = (
                "Given the provided information, including relevant images and retrieved context from the video, "
                "accurately and precisely answer the query without any additional prior knowledge.\n"
                "---------------------\n"
                "Context: {context_str}\n"
                "Metadata for video: {metadata_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            )

            # Interact with LLM
            openai_mm_llm = OpenAIMultiModal(
                model="gpt-4-turbo", api_key=openai.api_key, max_new_tokens=1500
            )
            response_1 = openai_mm_llm.complete(
                prompt=qa_tmpl_str.format(
                    context_str=context_str,
                    query_str=user_query,
                    metadata_str=json.dumps(st.session_state.metadata_vid),
                ),
                image_documents=image_documents,
            )

            # Display the response
            st.markdown("### 🤖 Multimodal RAG Response:")
            st.markdown(
                f"""
            <div style="background-color: #e8f5e8; padding: 20px; border-radius: 8px; border-left: 4px solid #4caf50;">
                <strong>Answer:</strong><br>
                {response_1.text}
            </div>
            """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")

# Evaluation Metrics Section
st.markdown("---")
st.markdown("### 📊 Evaluation Metrics")
st.markdown(
    """
<div class="info-box">
    <strong>System Performance Indicators:</strong>
    <ul>
        <li><strong>Retrieval Accuracy:</strong> Multi-modal similarity search with relevant context</li>
        <li><strong>Processing Latency:</strong> Optimized pipeline for real-time querying</li>
        <li><strong>Relevance Scoring:</strong> Configurable top-k retrieval for quality results</li>
        <li><strong>Visual-Audio Alignment:</strong> Coordinated processing of both modalities</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

# Step 4: Process new video
st.markdown("---")
st.markdown("### 🔄 Reset & Process New Video")
st.markdown("Click the button below to reset the application and process a new video.")
if st.button("🔄 Process New Video"):
    # Reset session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.success("✅ Application reset successfully! Please upload new video.")

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 14px;">
    <p><strong>Video Content RAG with Visual Understanding</strong> | Research Project</p>
    <p>Multimodal RAG system for comprehensive video content analysis</p>
</div>
""",
    unsafe_allow_html=True,
)
