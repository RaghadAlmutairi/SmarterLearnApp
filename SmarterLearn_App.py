import streamlit as st
import os
import yt_dlp
import re
import graphviz
from pydub import AudioSegment
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from PIL import Image
from openai import OpenAI
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from functools import lru_cache

# Set page configuration
st.set_page_config(page_title="SmarterLearn", page_icon="🎓", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #ff7e5f, #feb47b); /* Gradient background */
    }
    .big-font {
        font-size: 45px !important;
        color: #fff;
        font-weight: bold;
    }
    .sub-header {
        font-size: 24px !important;
        color: #fff;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #fff;
        color: #ff7e5f;
        font-size: 18px;
        border-radius: 50px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #feb47b;
        color: #fff;
    }
    .stTextInput>div>input {
        font-size: 16px !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024
AUDIO_PART_DURATION_MS = 2 * 60 * 1000
COLLECTION_NAME = "video_transcripts"
CACHE_DIR = "cache"

# Initialize Session State
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'cleaned_text' not in st.session_state:
    st.session_state.cleaned_text = None
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = None
if 'rag_config' not in st.session_state:
    st.session_state.rag_config = None
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_style' not in st.session_state:
    st.session_state.current_style = None
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ""
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_index' not in st.session_state:
    st.session_state.quiz_index = 0
if 'cache_key' not in st.session_state:
    st.session_state.cache_key = None

# Environment Setup
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name=COLLECTION_NAME,
    persist_directory="./chroma_db"
)

# Ensure cache directory exists at startup
os.makedirs(CACHE_DIR, exist_ok=True)

# Utility Functions
def clean_youtube_url(url):
    """Clean and standardize YouTube URL."""
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url.split("&")[0]

@lru_cache(maxsize=1)
def download_audio_from_youtube(url, output_path='downloads'):
    """Download audio from a YouTube video and convert it to WAV format."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'cookiesfrombrowser': ('chrome',),
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'retries': 3,
        'fragment_retries': 3,
        'ignoreerrors': False,
    }
    os.makedirs(output_path, exist_ok=True)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.wav'
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Downloaded audio file not found: {filename}")
        return os.path.abspath(filename)
    except Exception as e:
        raise Exception(f"Failed to download audio: {e}")

def split_audio(audio_path):
    """Split an audio file into smaller parts to comply with size limits."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        raise Exception(f"Failed to load audio file {audio_path}: {e}")

    total_duration_ms = len(audio)
    parts = []
    start = 0
    part_count = 0
    output_dir = os.path.dirname(audio_path)
    
    while start < total_duration_ms:
        end = min(start + AUDIO_PART_DURATION_MS, total_duration_ms)
        part = audio[start:end]
        
        temp_filename = os.path.join(output_dir, f"temp_part_{part_count}.wav")
        try:
            part.export(temp_filename, format="wav")
            file_size = os.path.getsize(temp_filename)
            if file_size > MAX_AUDIO_SIZE_BYTES:
                os.remove(temp_filename)
                raise ValueError(f"Part {part_count} exceeds size limit of {MAX_AUDIO_SIZE_BYTES} bytes")
            
            final_filename = os.path.join(output_dir, f"part_{part_count}.wav")
            os.rename(temp_filename, final_filename)
            parts.append(os.path.abspath(final_filename))
            start = end
            part_count += 1
        except Exception as e:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise Exception(f"Failed to export audio part {part_count}: {e}")
    
    return parts

def transcribe_audio_openai(audio_path):
    """Transcribe audio using OpenAI's Whisper model."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

def clean_transcript(text):
    """Clean and format transcript text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace('. ', '.\n').strip()
    return text

def generate_cache_key(url):
    """Generate a unique cache key based on the URL."""
    return hashlib.md5(url.encode()).hexdigest()

@lru_cache(maxsize=128)
def preprocess_youtube_video(url):
    """Preprocess a YouTube video by downloading audio, splitting it, and transcribing."""
    try:
        clean_url = clean_youtube_url(url)
        audio_path = download_audio_from_youtube(clean_url)
        split_files = split_audio(audio_path)
        all_text = []
        for part_file in split_files:
            if os.path.exists(part_file):
                text = transcribe_audio_openai(part_file)
                all_text.append(text)
                try:
                    os.remove(part_file)
                except Exception as e:
                    st.warning(f"Failed to delete part file {part_file}: {e}")
            else:
                raise FileNotFoundError(f"Split audio file not found: {part_file}")
        
        transcript_text = "\n".join([text for text in all_text if text.strip()])
        if not transcript_text.strip():
            raise ValueError("Transcript is empty. Please try a different video.")
        
        cleaned_text = clean_transcript(transcript_text)
        try:
            os.remove(audio_path)
        except Exception as e:
            st.warning(f"Failed to delete audio file {audio_path}: {e}")
        return cleaned_text
    except Exception as e:
        raise Exception(f"Preprocessing failed: {e}")

# Visual Learning (Mind Map)
@lru_cache(maxsize=128)
def extract_mindmap(cleaned_text):
    """Extract a mind map structure from the transcript."""
    if not cleaned_text.strip():
        return "- [NO CONTENT]"
    prompt = f"""
You are an expert in summarization and mind mapping.

Here is a cleaned transcription of a YouTube educational video:

{cleaned_text}

Your task:
1. Identify the overall MAIN TOPIC.
2. Extract key branches and subpoints in a hierarchy.

⚠️ Follow this format:
- [MAIN TOPIC]
    - Main Branch 1
        - Sub-idea 1
        - Sub-idea 2
    - Main Branch 2
        - Sub-idea 1
        - Sub-idea 2

Output only the structure above. No extra commentary.
"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in summarization."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

def parse_outline_to_edges(outline_text):
    """Parse mind map outline into graph edges."""
    lines = outline_text.strip().split('\n')
    edges = []
    parent_stack = []
    root = None
    for line in lines:
        match = re.match(r'^(\s*)-?\s*(.+)$', line)
        if match:
            indent = len(match.group(1))
            content = match.group(2).strip()
            if root is None:
                root = content
                parent_stack.append((indent, root))
                continue
            while parent_stack and indent <= parent_stack[-1][0]:
                parent_stack.pop()
            if parent_stack:
                parent = parent_stack[-1][1]
                edges.append((parent, content))
            parent_stack.append((indent, content))
    return edges, root

def generate_distinct_pastel_colors(n):
    """Generate distinct pastel colors for mind map nodes."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        lightness = 0.85
        saturation = 0.4
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

@lru_cache(maxsize=128)
def generate_mindmap_image(outline_text, output_filename="mindmap_generated"):
    """Generate a mind map image from the outline."""
    dot = graphviz.Digraph(comment="Mind Map", engine="twopi")
    dot.graph_attr.update(splines="curved", nodesep="1.0", ranksep="2.0", overlap="false")
    edges, root = parse_outline_to_edges(outline_text)
    node_hierarchy = {}
    for parent, child in edges:
        if parent not in node_hierarchy:
            node_hierarchy[parent] = []
        node_hierarchy[parent].append(child)
    color_map = {root: "#F7E1A1"}
    top_nodes = node_hierarchy.get(root, [])
    pastel_colors = generate_distinct_pastel_colors(len(top_nodes))
    for node, color in zip(top_nodes, pastel_colors):
        color_map[node] = color
    def propagate_colors(node, parent_color):
        if node not in color_map:
            color_map[node] = parent_color
        if node in node_hierarchy:
            for child in node_hierarchy[node]:
                propagate_colors(child, color_map[node])
    for node in top_nodes:
        propagate_colors(node, color_map[node])
    for parent, child in edges:
        dot.node(parent, shape="box", style="filled,setlinewidth(2)", fillcolor=color_map.get(parent, "#D3D3D3"), fontcolor="black", fontsize="12")
        dot.node(child, shape="box", style="filled,setlinewidth(2)", fillcolor=color_map.get(child, "#D3D3D3"), fontcolor="black", fontsize="12")
        dot.edge(parent, child, color=color_map.get(parent, "#D3D3D3"))
    dot.render(output_filename, format='png', cleanup=True)
    return output_filename + ".png"

# Kinesthetic Learning (Quiz)
@lru_cache(maxsize=128)
def generate_quiz(cleaned_text):
    """Generate a quiz from the transcript."""
    prompt = f"""
You are an expert educator.
Based on the transcript below, generate exactly 5 multiple-choice quiz questions:
Transcript:
{cleaned_text[:2000]}...
Requirements:
- Each question must have exactly 4 answer choices labeled A, B, C, D.
- Write the correct answer like: Correct: B
- Return plain text in the format:
Question?
A) Option A
B) Option B
C) Option C
D) Option D
Correct: X
"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a quiz creator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

# RAG Setup and Tools
def split_text_into_chunks(text):
    """Split text into chunks for vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def store_chunks_in_vectorstore(chunks):
    """Store text chunks in the vector store."""
    documents = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]
    if not documents:
        raise ValueError("No valid text chunks to embed. Ensure the transcript contains usable content.")
    vector_store.add_documents(documents)

@tool
def retrieve(query: str) -> str:
    """Retrieve relevant content chunks from the vector store based on the user query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    if not retrieved_docs:
        all_docs = vector_store.similarity_search("summary of the video", k=3)
        return "Summary of the video transcript:\n\n" + "\n\n".join((f"Content: {doc.page_content}") for doc in all_docs)
    return "\n\n".join((f"Content: {doc.page_content}") for doc in retrieved_docs)

@tool
def summarize_transcript(transcript: str) -> str:
    """Summarize the entire transcript into a concise paragraph."""
    prompt = f"""
Summarize the following transcript into a concise paragraph (3-5 sentences) capturing the main ideas:

{transcript}

Output only the summary paragraph, no additional commentary.
"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in summarization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error summarizing transcript: {e}")
        return f"Error summarizing transcript: {e}"

@tool
def generate_key_points(transcript: str) -> str:
    """Extract key points from the transcript as bullet points."""
    prompt = f"""
Extract 5-7 key points from the following transcript as bullet points:

{transcript}

Format:
- Key point 1
- Key point 2
- ...

Output only the bullet points, no additional commentary.
"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in summarization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating key points: {e}")
        return f"Error generating key points: {e}"

@tool
def define_term(term: str, transcript: str) -> str:
    """Define a specific term or concept mentioned in the transcript or query."""
    prompt = f"""
Based on the following transcript, provide a clear and concise definition of the term "{term}".
If the term is not explicitly mentioned, provide a general definition relevant to the context of the transcript.

Transcript:
{transcript[:2000]}...

Output only the definition, no additional commentary.
"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in education and definitions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error defining term: {e}")
        return f"Error defining term: {e}"

@tool
def compare_concepts(concept1: str, concept2: str, transcript: str) -> str:
    """Compare two concepts mentioned in the transcript."""
    prompt = f"""
Based on the following transcript, compare the concepts "{concept1}" and "{concept2}".
Provide a concise comparison highlighting their similarities and differences.

Transcript:
{transcript[:2000]}...

Output only the comparison, no additional commentary.
"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in educational comparisons."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error comparing concepts: {e}")
        return f"Error comparing concepts: {e}"

def setup_rag(cleaned_text):
    """Set up RAG agent for answering questions."""
    chunks = split_text_into_chunks(cleaned_text)
    store_chunks_in_vectorstore(chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    memory = MemorySaver()
    tools = [retrieve, summarize_transcript, generate_key_points, define_term, compare_concepts]
    rag_agent = create_react_agent(llm, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "session_01"}}
    return rag_agent, config

def get_assistant_response(user_text, rag_agent, config, transcript_text):
    """Get a response from the RAG agent with optimized tool usage."""
    system_prompt = """
You are a helpful AI tutor. Answer questions strictly based on the provided YouTube video transcript. 
Use the available tools (retrieve, summarize_transcript, generate_key_points, define_term, compare_concepts) to provide accurate and relevant answers.
Pass the transcript as an argument to tools that require it.
If the question is unrelated to the transcript, respond only with: 
"Sorry, I can only answer questions related to the video content. Please ask something about the video."
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript: {transcript_text}\n\nUser question: {user_text}"}
    ]
    try:
        response = rag_agent.invoke({"messages": messages}, config=config)
        answer = response['messages'][-1].content

        # Similarity check to ensure relevance
        answer_embedding = embedding_function.embed_query(answer)
        transcript_chunks = split_text_into_chunks(transcript_text)
        transcript_embeddings = embedding_function.embed_documents(transcript_chunks)
        similarities = [
            np.dot(answer_embedding, chunk_embedding) / 
            (np.linalg.norm(answer_embedding) * np.linalg.norm(chunk_embedding))
            for chunk_embedding in transcript_embeddings
        ]
        similarity_threshold = 0.7
        if max(similarities, default=0) < similarity_threshold:
            return "Sorry, I can only answer questions related to the video content. Please ask something about the video."
        return answer
    except Exception as e:
        st.error(f"RAG error: {e}")
        return "Sorry, I couldn't process your request. Please try again."

# Auditory Learning
def transcribe_user_audio(audio_file):
    """Transcribe an uploaded audio file."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def speak_response(text, filename="reply.mp3"):
    """Generate speech from text and save as an MP3 file."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        speech = client.audio.speech.create(model="tts-1", voice="nova", input=text)
        with open(filename, "wb") as f:
            f.write(speech.content)
        return filename
    except Exception as e:
        st.error(f"Speech generation error: {e}")
        return None

# Smart Loop Suggestion
def suggest_learning_style(current_style):
    """Suggest a different learning style based on the current one."""
    styles = {
        'visual': ("Would you like to try a mind map to visualize the video content? It’s great for seeing the big picture!", "visual"),
        'reading': ("How about exploring the video with interactive Q&A? You can ask questions and get detailed answers!", "reading"),
        'kinesthetic': ("Why not test your knowledge with a quiz? It’s a fun way to engage with the material!", "kinesthetic"),
        'auditory': ("Want to try voice-based learning? You can ask questions and hear the answers!", "auditory")
    }
    recommendations = {
        "kinesthetic": "auditory",
        "auditory": "kinesthetic",
        "reading": "visual",
        "visual": "reading"
    }
    suggested_style = recommendations.get(current_style, 'reading')
    return styles[suggested_style]

# Home Page
def show_homepage():
    # Title and Description
    st.title("Welcome to SmarterLearn", anchor="top")
    st.markdown("""
        **SmarterLearn** uses AI tools to make learning more efficient based on your unique learning style.
        Whether you're a visual, auditory, or kinesthetic learner, we help you learn smarter, not harder.
        Find out your learning type and start your personalized learning journey!
    """)

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Start Learning'):
            st.session_state.page = "learning"

    with col2:
        if st.button('Discover Your Learning Type'):
            st.session_state.page = "quiz"

    # Section for video
    st.markdown("<div class='sub-header'>Learn Smarter with Video</div>", unsafe_allow_html=True)
    st.markdown("""
        Watch educational videos re-imagined to match your learning type. 
        Our AI highlights key moments in the video that suit your learning style.
    """)

    # Embed a YouTube video
    st.video("https://youtu.be/_IopcOwfsoU?si=JU4mjxCYWrmYGTLe")  # Updated YouTube video URL

# Quiz Page
def show_quiz():
    # Initialize scores for each learning type
    learning_types = {
        "Visual": 0,
        "Auditory": 0,
        "Kinesthetic": 0,
        "Reading/Writing": 0
    }

    # Quiz questions
    questions = [
        {
            "question": "When you learn something new, what helps you the most?",
            "options": ["Seeing diagrams and charts", "Listening to explanations", "Doing hands-on activities", "Reading about it"]
        },
        {
            "question": "How do you usually study for an exam?",
            "options": ["I draw mind maps", "I listen to audio recordings", "I practice by doing exercises", "I take detailed notes and rewrite them"]
        },
        {
            "question": "Which of the following do you enjoy the most?",
            "options": ["Watching videos", "Listening to podcasts", "Playing sports", "Writing essays or notes"]
        },
        {
            "question": "How do you best remember new information?",
            "options": ["By picturing it in my mind", "By repeating it out loud", "By trying it myself", "By writing it down over and over"]
        },
        {
            "question": "What is your favorite way of learning new skills?",
            "options": ["Watching instructional videos", "Listening to lectures", "Hands-on practice", "Reading manuals and instructions"]
        }
    ]

    # Function to display the quiz and collect answers
    def ask_questions():
        for q in questions:
            st.subheader(q["question"])
            answer = st.radio("Choose your answer:", q["options"], key=q["question"])
            if answer:
                # Update the scores based on the answer
                if answer == q["options"][0]:  # Visual
                    learning_types["Visual"] += 1
                elif answer == q["options"][1]:  # Auditory
                    learning_types["Auditory"] += 1
                elif answer == q["options"][2]:  # Kinesthetic
                    learning_types["Kinesthetic"] += 1
                elif answer == q["options"][3]:  # Reading/Writing
                    learning_types["Reading/Writing"] += 1

    # Start quiz
    ask_questions()

    # Calculate the percentage for each learning type
    total_answers = sum(learning_types.values())
    if total_answers > 0:
        for key in learning_types:
            learning_types[key] = (learning_types[key] / total_answers) * 100

        # Display a histogram of the results
        st.subheader("Your Learning Style Breakdown")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(learning_types.keys(), learning_types.values(), color=["#ff7e5f", "#feb47b", "#74b9ff", "#55efc4"])
        ax.set_xlabel("Learning Types")
        ax.set_ylabel("Percentage")
        ax.set_title("Your Learning Style Preferences")
        st.pyplot(fig)

        # Display the result with a funny twist
        st.subheader("Your Best Learning Type:")
        best_type = max(learning_types, key=learning_types.get)
        percentage = learning_types[best_type]

        if best_type == "Visual":
            st.markdown(f"🎨 You are a **Visual Learner**! You learn best when you can see the information. Diagrams, pictures, and videos are your best friends!")
        elif best_type == "Auditory":
            st.markdown(f"🎧 You are an **Auditory Learner**! You learn best by listening. Podcasts, explanations, and group discussions are your jam!")
        elif best_type == "Kinesthetic":
            st.markdown(f"🏃‍♂️ You are a **Kinesthetic Learner**! You learn by doing! Hands-on activities, experiments, and movement are your secret sauce!")
        elif best_type == "Reading/Writing":
            st.markdown(f"📚 You are a **Reading/Writing Learner**! You learn best by reading and writing. Notes, essays, and textbooks are your happy place!")

        # Fun closing message
        st.markdown(f"Overall, you have {best_type} traits with {round(percentage, 2)}% dominance. Remember, learning is fun, so embrace it with your unique style! 🎉")
    else:
        st.warning("Please answer all the questions to get your result!")

    # Add a button to go back to the homepage
    if st.button("Back to Homepage"):
        st.session_state.page = "home"




# Start Learning Page
def show_learning_page():
    # Title and description
    st.title("Start Learning Your Way!")
    st.markdown("""
        Paste a YouTube video URL below, and choose the best learning style for you. 
        Based on your learning style, we’ll provide suggestions for how you can make the most of this video!
    """)

    # Input field for YouTube video URL
    video_url = st.text_input("Paste your YouTube video URL:", value=st.session_state.youtube_url)

    # Display the video if URL is entered
    if video_url:
        st.video(video_url)

    # Process the video if not already processed
    if video_url and (not st.session_state.processed or st.session_state.youtube_url != video_url):
        with st.spinner("Processing video..."):
            try:
                cache_key = generate_cache_key(video_url)
                st.session_state.cache_key = cache_key
                cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
                
                # Attempt to read from cache first
                if os.path.exists(cache_file):
                    with open(cache_file, "r", encoding="utf-8") as f:
                        st.session_state.cleaned_text = f.read()
                    st.session_state.processed = True
                else:
                    # Process the video and save to cache
                    st.session_state.cleaned_text = preprocess_youtube_video(video_url)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(st.session_state.cleaned_text)
                    st.session_state.processed = True

                st.session_state.youtube_url = video_url
                if not st.session_state.rag_agent:
                    st.session_state.rag_agent, st.session_state.rag_config = setup_rag(st.session_state.cleaned_text)
                st.success("Video processed successfully!")
            except Exception as e:
                st.error(f"Error processing video: {e}")
                st.session_state.processed = False
                st.session_state.cleaned_text = None

    # Buttons for the different learning types
    if st.session_state.processed and not st.session_state.current_style:
        st.markdown("<div class='sub-header'>Select Your Learning Style</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # Visual Learner button
        with col1:
            if st.button('👀 Visual Learner'):
                st.session_state.current_style = "visual"

        # Auditory Learner button
        with col2:
            if st.button('🎧 Auditory Learner'):
                st.session_state.current_style = "auditory"

        # Kinesthetic Learner button
        with col1:
            if st.button('🏃‍♂️ Kinesthetic Learner'):
                st.session_state.current_style = "kinesthetic"

        # Reading/Writing Learner button
        with col2:
            if st.button('📚 Reading/Writing Learner'):
                st.session_state.current_style = "reading"

    # Display the selected learning style's content
    if st.session_state.processed and st.session_state.current_style:
        # Visual Learner
        if st.session_state.current_style == "visual":
            with st.spinner("Generating mind map..."):
                outline = extract_mindmap(st.session_state.cleaned_text)
                mindmap_path = generate_mindmap_image(outline)
                st.image(Image.open(mindmap_path), caption="Mind Map of Video Content")

        # Auditory Learner
        elif st.session_state.current_style == "auditory":
            st.markdown("""
                **Ask a Question with Your Voice**:
                - Record your question using a voice recorder (e.g., on your phone or computer) as a WAV or MP3 file.
                - Upload the audio file below to get a spoken response.
            """)
            uploaded_audio = st.file_uploader("Upload your recorded question (WAV or MP3):", type=["wav", "mp3"], key="auditory_uploader")
            if uploaded_audio:
                with st.spinner("Processing your audio..."):
                    user_text = transcribe_user_audio(uploaded_audio)
                    if user_text:
                        st.write(f"**You said:** {user_text}")
                        response = get_assistant_response(
                            user_text, 
                            st.session_state.rag_agent, 
                            st.session_state.rag_config, 
                            st.session_state.cleaned_text
                        )
                        st.write(f"**AI Response:** {response}")
                        audio_response_path = speak_response(response)
                        if audio_response_path:
                            st.audio(audio_response_path)
                        st.session_state.conversation_history.append((user_text, response))

            # Display conversation history
            if st.session_state.conversation_history:
                st.subheader("Conversation History")
                for idx, (question, answer) in enumerate(st.session_state.conversation_history, 1):
                    st.markdown(f"**Q{idx}:** {question}")
                    st.markdown(f"**A{idx}:** {answer}")
                    st.markdown("---")

        # Kinesthetic Learner
        elif st.session_state.current_style == "kinesthetic":
            if not st.session_state.quiz_questions:
                quiz_text = generate_quiz(st.session_state.cleaned_text)
                st.session_state.quiz_questions = quiz_text.strip().split('\n\n')
                st.session_state.quiz_score = 0
                st.session_state.quiz_index = 0

            if st.session_state.quiz_index < len(st.session_state.quiz_questions):
                q = st.session_state.quiz_questions[st.session_state.quiz_index]
                lines = q.strip().split('\n')
                question = lines[0]
                choices = lines[1:5]
                correct_line = [l for l in lines if l.lower().startswith("correct")]
                correct_answer = correct_line[0].split(":")[1].strip().upper() if correct_line else None

                st.subheader(f"Question {st.session_state.quiz_index + 1}/{len(st.session_state.quiz_questions)}")
                st.write(question)
                # Use a unique key for each question to prevent widget state issues
                user_answer = st.radio("Select your answer:", choices, key=f"quiz_{st.session_state.quiz_index}_{st.session_state.cache_key}")

                if st.button("Submit Answer", key=f"submit_quiz_{st.session_state.quiz_index}_{st.session_state.cache_key}"):
                    selected_letter = user_answer.split(')')[0]
                    if selected_letter == correct_answer:
                        st.success("Correct!")
                        st.session_state.quiz_score += 1
                    else:
                        st.error(f"Wrong. The correct answer was: {correct_answer}")
                    st.session_state.quiz_index += 1
            else:
                st.subheader("Quiz Complete!")
                st.write(f"Your Score: {st.session_state.quiz_score}/{len(st.session_state.quiz_questions)}")
                if st.button("Restart Quiz", key="restart_quiz"):
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_index = 0

        # Reading/Writing Learner
        elif st.session_state.current_style == "reading":
            # Display conversation history
            if st.session_state.conversation_history:
                st.subheader("Conversation History")
                for idx, (question, answer) in enumerate(st.session_state.conversation_history, 1):
                    st.markdown(f"**Q{idx}:** {question}")
                    st.markdown(f"**A{idx}:** {answer}")
                    st.markdown("---")

            # Form for asking questions
            with st.form(key="question_form"):
                question_input = st.text_input("Ask a question about the video (e.g., 'What is the main topic?', 'Summarize the video'):", value=st.session_state.question, key="question_input")
                submit_button = st.form_submit_button("Submit Question")
                
                if submit_button and question_input:
                    with st.spinner("Thinking..."):
                        try:
                            response = get_assistant_response(
                                question_input, 
                                st.session_state.rag_agent, 
                                st.session_state.rag_config, 
                                st.session_state.cleaned_text
                            )
                            st.session_state.conversation_history.append((question_input, response))
                            st.session_state.question = ""
                        except Exception as e:
                            st.error(f"Error answering question: {e}")

        # Smart Loop Suggestion
        suggestion, suggested_style = suggest_learning_style(st.session_state.current_style)
        st.markdown(f"🌟 **Suggestion:** {suggestion}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Try {suggested_style} style with the same video"):
                st.session_state.current_style = suggested_style
                st.session_state.quiz_questions = []  # Reset quiz if switching styles
                st.rerun()
        with col2:
            if st.button("Choose a different learning style"):
                st.session_state.current_style = None
                st.session_state.quiz_questions = []
                st.rerun()

    # Add Back to Homepage button
    if st.button("Back to Homepage"):
        st.session_state.page = "home"
        st.rerun()

    # Option to Start a New Session
    if st.session_state.processed:
        if st.button("Start a New Session with a Different Video"):
            # Reset session state
            st.session_state.processed = False
            st.session_state.cleaned_text = None
            st.session_state.audio_path = None
            st.session_state.rag_agent = None
            st.session_state.rag_config = None
            st.session_state.question = ""
            st.session_state.conversation_history = []
            st.session_state.current_style = None
            st.session_state.youtube_url = ""
            st.session_state.quiz_questions = []
            st.session_state.quiz_score = 0
            st.session_state.quiz_index = 0
            st.session_state.cache_key = None
            st.rerun()

# Main logic to switch between pages
if "page" not in st.session_state:
    st.session_state.page = "home"  # Default to homepage

# Display the page based on current session state
if st.session_state.page == "home":
    show_homepage()
elif st.session_state.page == "quiz":
    show_quiz()
else:
    show_learning_page()            