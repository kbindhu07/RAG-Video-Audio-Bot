# 🎬 Video Content RAG with Visual Understanding

A multimodal RAG (Retrieval-Augmented Generation) system that processes video content, extracts key frames, transcribes audio, and allows users to query both visual and audio elements from video libraries.

## 🎯 Problem Statement

Create a multimodal RAG system that processes video content, extracts key frames, transcribes audio, and allows users to query both visual and audio elements from video libraries.

## ✨ Key Features

- **Video Processing & Key Frame Extraction**: Automatically extracts key visual frames from videos at optimal intervals
- **Audio Transcription & Synchronization**: Converts video audio to text with Whisper for comprehensive content understanding
- **Visual Element Recognition**: Implements CLIP-based visual understanding for scene and object detection
- **Multi-modal Search Capabilities**: Combines visual and audio elements for comprehensive video querying
- **Temporal Indexing**: Maintains timestamp references for precise video content retrieval
- **Vector Database Integration**: Uses LanceDB for efficient storage and retrieval of multimodal embeddings

## 🏗️ Technical Architecture

### Core Components

1. **Video Preprocessing Pipeline**
   - YouTube video download and processing
   - Key frame extraction at 0.5 FPS for optimal analysis
   - Audio extraction and conversion to WAV format

2. **Multi-modal Embedding System**
   - CLIP embeddings for visual content understanding
   - OpenAI Whisper for audio transcription
   - OpenAI GPT-4 Turbo for multimodal reasoning

3. **Vector Database & Retrieval**
   - LanceDB for efficient storage of text and image embeddings
   - Multi-modal similarity search with configurable top-k retrieval
   - Separate collections for text and visual content

4. **Query Processing & Generation**
   - Context-aware prompt engineering
   - Multi-modal LLM integration for comprehensive responses
   - Temporal context preservation

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- FFmpeg (for video processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-content-rag.git
cd video-content-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Usage

1. **Run the Streamlit Application**:
```bash
streamlit run app.py
```

2. **Access the Web Interface**:
   - Open your browser and navigate to `http://localhost:8501`
   - Enter your OpenAI API key
   - Provide a YouTube video URL
   - Process the video to extract audio and visual frames
   - Ask questions about the video content

## 📊 Technical Implementation

### Video Processing Pipeline

```python
# Key frame extraction at 0.5 FPS
def video_to_images(video_path, output_folder):
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.5)
```

### Multi-modal Indexing

```python
# Separate vector stores for text and images
text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")

# Multi-modal index creation
index = MultiModalVectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

### Retrieval System

```python
# Multi-modal retrieval with configurable top-k
retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)
```

## 🎯 Key Requirements Met

- ✅ **Video processing and key frame extraction**: Implemented with MoviePy
- ✅ **Audio transcription and synchronization**: Whisper-based transcription
- ✅ **Visual element recognition and tagging**: CLIP embeddings for visual understanding
- ✅ **Multi-modal search capabilities**: Combined visual + audio retrieval
- ✅ **Temporal indexing with timestamp referencing**: Frame-based temporal context

## 🔧 Technical Challenges Addressed

- **Video preprocessing and frame selection algorithms**: Optimized frame extraction at 0.5 FPS
- **Visual-audio synchronization and alignment**: Coordinated processing pipeline
- **Object detection and scene understanding**: CLIP-based visual analysis
- **Multi-modal embedding space creation**: Unified embedding strategy
- **Large-scale video data storage and indexing**: LanceDB vector database

## 📈 Evaluation Metrics

The system provides:
- **Retrieval Accuracy**: Multi-modal similarity search with relevant context
- **Latency**: Optimized processing pipeline for real-time querying
- **Relevance Scoring**: Configurable top-k retrieval for quality results

## 🛠️ Technology Stack

- **Frontend**: Streamlit for interactive web interface
- **Video Processing**: MoviePy, PyTubeFix
- **Audio Processing**: SpeechRecognition, Whisper
- **Computer Vision**: CLIP, PIL, scikit-image
- **Vector Database**: LanceDB
- **LLM Integration**: OpenAI GPT-4 Turbo
- **Embeddings**: CLIP for visual, OpenAI for text

## 📁 Project Structure

```
video-content-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── mixed_data/           # Processed video data
│   ├── output_text.txt   # Transcribed audio
│   └── frame*.png        # Extracted key frames
└── video_data/           # Downloaded video files
    └── input_vid.mp4     # Processed video
```

## 🚀 Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy with the following configuration:
   - Python version: 3.9+
   - Dependencies: `requirements.txt`

### Environment Variables

Set the following environment variables in your deployment:
- `OPENAI_API_KEY`: Your OpenAI API key




## 🙏 Acknowledgments

- OpenAI for GPT-4-turbo multimodal capabilities
- LlamaIndex team for the excellent RAG framework
- Streamlit for the beautiful web interface
- The open-source community for various supporting libraries

---

**Ready to turn any YouTube video into an interactive AI assistant? Start exploring now! 🚀**
