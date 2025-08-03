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

# Streamlit app setup with a custom layout and title
st.set_page_config(
    page_title="YouTube Video Q&A with LLM", layout="centered", page_icon="🎬"
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 YouTube Video Q&A with LLM")
st.markdown("---")
st.subheader(
    "🚀 Process and ask questions about YouTube videos effortlessly! Helpful for studying!"
)

st.markdown(
    """
    **How it works:**
    1. Enter your OpenAI API key
    2. Provide a YouTube video URL
    3. Process the video to extract audio and frames
    4. Ask questions about the video content
    
    The app will analyze both the audio transcript and visual frames to provide comprehensive answers!
    """
)
st.markdown("---")

# OpenAI API key input
st.markdown("### 🔑 Step 1: Enter your OpenAI API Key")
st.markdown(
    "You'll need an OpenAI API key to use this application. Get one from [OpenAI's platform](https://platform.openai.com/api-keys)."
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
    yt = YouTube(url)
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(
        output_path=output_path, filename="input_vid.mp4"
    )
    return metadata


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

# Step 2: Input YouTube URL
st.markdown("### 🎥 Step 2: Enter the YouTube Video URL")
st.markdown(
    "Provide a valid YouTube video URL. The app will download and process the video to extract audio and visual frames."
)
video_url = st.text_input(
    "Enter the YouTube video link:",
    key="video_input",
    placeholder="https://www.youtube.com/watch?v=example",
    help="Paste a complete YouTube URL here. The video will be downloaded and processed.",
)

# Process the video on button click
if video_url and st.session_state.retriever_engine is None:
    if st.button("🚀 Process Video"):
        try:
            with st.spinner("Processing video... This might take a while :( "):
                st.session_state.metadata_vid = download_video(
                    video_url, output_video_path
                )
                video_to_images(filepath, output_folder)
                video_to_audio(filepath, output_audio_path)
                text_data = audio_to_text(output_audio_path)

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

                st.success("✅ Video processing completed! You can now ask questions.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Step 3: Ask questions
if st.session_state.retriever_engine:
    st.markdown("### 💬 Step 3: Ask a Question About the Video")
    st.markdown(
        "Now you can ask questions about the video content. The app will analyze both the audio transcript and visual frames to provide answers."
    )
    user_query = st.text_input(
        "Ask a question about the video:",
        key="question_input",
        placeholder="What is the main topic discussed in this video?",
        help="Ask any question about the video content, topics, or details shown in the video.",
    )

    if st.button("🔍 Submit Query") and user_query:
        try:
            img, txt = retrieve(
                retriever_engine=st.session_state.retriever_engine, query_str=user_query
            )
            image_documents = SimpleDirectoryReader(
                input_dir=output_folder, input_files=img
            ).load_data()
            context_str = "".join(txt)

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
            st.markdown("### 🤖 LLM Response:")
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
            st.error(f"An error occurred: {e}")

# Step 4: Process new video
st.markdown("---")
st.markdown("### 🔄 Start Over")
st.markdown("Click the button below to reset the application and process a new video.")
if st.button("🔄 Process New Video"):
    # Reset session state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.success("✅ Application reset successfully! Please enter a new video link.")
