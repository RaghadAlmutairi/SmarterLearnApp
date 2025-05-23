# -*- coding: utf-8 -*-
"""SmarterLearn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hw5rGX2_cb5kRsO50vcpgxeFmqw_f-MK
"""
import os
import yt_dlp
import re
import openai
import graphviz
from pydub import AudioSegment
from IPython.display import Image, display, Audio
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from playsound import playsound
import time
import pyaudio
import wave
from dotenv import load_dotenv
import colorsys  # Added for pastel color generation
import numpy as np
import requests

# Constants
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024
AUDIO_PART_DURATION_MS = 2 * 60 * 1000
COLLECTION_NAME = "video_transcripts"
env_path = r'C:\Users\rkhm3\Desktop\PythonProjectsTasks\EduTool_project\.env'
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print("❌ .env file not found at:", env_path)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
print("✅ OPENAI Key loaded:", OPENAI_API_KEY is not None)
print("✅ HuggingFace Key loaded:", HUGGINGFACEHUB_API_TOKEN is not None)

# Initialize vector store
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(embedding_function=embedding_function, collection_name=COLLECTION_NAME, persist_directory="./chroma_db")

# Unified preprocessing function
def preprocess_youtube_video(url):
    try:
        clean_url = clean_youtube_url(url)
        print(f"✅ Cleaned URL: {clean_url}")
        audio_path = download_audio_from_youtube(clean_url)
        print(f"✅ Starting audio splitting for: {audio_path}")
        split_files = split_audio(audio_path)
        print(f"✅ Split into {len(split_files)} audio parts")
        all_text = []
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(transcribe_audio_openai, split_files))
            all_text.extend(results)
        for part_file in split_files:
            os.remove(part_file)
        transcript_text = "\n".join([text for text in all_text if text])
        print("✅ Starting transcript cleaning")
        cleaned_text = clean_transcript(transcript_text)
        print("✅ Transcript cleaned")
        return cleaned_text, audio_path
    except Exception as e:
        if "Unsupported URL" in str(e):
            raise Exception("Invalid URL: Please enter a valid YouTube URL (e.g., https://youtu.be/...).")
        print(f"❌ Preprocessing error: {e}")
        raise

# Clean YouTube URL
def clean_youtube_url(url):
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url.split("&")[0]

# Download YouTube Audio
def download_audio_from_youtube(url, output_path='./downloads'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
    }
    os.makedirs(output_path, exist_ok=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.wav'
    print(f"✅ Audio downloaded at: {filename}")
    return filename

# Split Audio into Parts
def split_audio(audio_path):
    print(f"📂 Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    print(f"📏 Audio duration: {total_duration_ms/1000:.2f} seconds")
    parts = []
    start = 0
    part_count = 0
    while start < total_duration_ms:
        end = min(start + AUDIO_PART_DURATION_MS, total_duration_ms)
        part = audio[start:end]
        print(f"✂️ Processing part {part_count+1} (start: {start/1000:.2f}s, end: {end/1000:.2f}s)")
        while len(part.export(format="wav").read()) > MAX_AUDIO_SIZE_BYTES and AUDIO_PART_DURATION_MS > 30 * 1000:
            part_duration_ms = AUDIO_PART_DURATION_MS // 2
            end = min(start + part_duration_ms, total_duration_ms)
            part = audio[start:end]
            print(f"🔄 Reduced part size to {part_duration_ms/1000:.2f}s")
        filename = f"./downloads/part_{part_count}.wav"
        part.export(filename, format="wav")
        print(f"✅ Saved part {part_count+1}: {filename}")
        parts.append(filename)
        start = end
        part_count += 1
    return parts

# Transcribe Audio using OpenAI Whisper
def transcribe_audio_openai(audio_path):
    print(f"📤 Sending {audio_path} to Whisper API")
    start_time = time.time()
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                timeout=60  # Add 60-second timeout
            )
        print(f"✅ Transcribed {audio_path} in {time.time() - start_time:.2f} seconds")
        return transcript.text
    except Exception as e:
        print(f"❌ Transcription error for {audio_path}: {e}")
        return ""

# Clean Transcript Text
def clean_transcript(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace('. ', '.\n').strip()
    return text

# Setup RAG components
def setup_rag(cleaned_text):
    chunks = split_text_into_chunks(cleaned_text)
    store_chunks_in_vectorstore(chunks)
    llm = init_chat_model(model='gpt-4o-mini', model_provider='openai')
    memory = MemorySaver()
    rag_agent = create_react_agent(llm, [retrieve], checkpointer=memory)
    config = {"configurable": {"thread_id": "session_01"}}
    return rag_agent, config

# Split Text into Chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_text(text)

# Store Chunks in Vector Store
def store_chunks_in_vectorstore(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store.add_documents(documents=documents)
    print(f"✅ {len(chunks)} chunks stored in vector database.")

# Retriever Tool
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    """Retrieve relevant content chunks from the vector store based on the user query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    if not retrieved_docs:
        all_docs = vector_store.similarity_search("summary of the video", k=3)
        serialized = "Summary of the video transcript:\n\n" + "\n\n".join((f"Content: {doc.page_content}") for doc in all_docs)
        return serialized, all_docs
    serialized = "\n\n".join((f"Content: {doc.page_content}") for doc in retrieved_docs)
    return serialized, retrieved_docs

# Enhanced Assistant Response
def get_assistant_response(user_text, rag_agent, config, transcript_text):
    system_prompt = """
You are a helpful AI tutor. Answer questions strictly based on the provided YouTube video transcript. 
If the question is unrelated to the transcript, respond only with: 
"Sorry, I can only answer questions related to the video content. Please ask something about the video."
Use the retrieved chunks or transcript summary to provide accurate answers.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript summary:\n{transcript_text[:1000]}...\n\nUser question: {user_text}"}
    ]
    try:
        response = rag_agent.invoke({"messages": messages}, config=config)
        answer = response['messages'][-1].content

        # Check if the answer is the out-of-scope message
        if answer == "Sorry, I can only answer questions related to the video content. Please ask something about the video.":
            return answer

        # Generate embeddings for answer and transcript
        answer_embedding = embedding_function.embed_query(answer)
        transcript_chunks = split_text_into_chunks(transcript_text)  # Reuse existing chunking function
        transcript_embeddings = embedding_function.embed_documents(transcript_chunks)

        # Compute cosine similarities between answer and transcript chunks
        similarities = [
            np.dot(answer_embedding, chunk_embedding) / 
            (np.linalg.norm(answer_embedding) * np.linalg.norm(chunk_embedding))
            for chunk_embedding in transcript_embeddings
        ]

        # Check if max similarity is above a threshold
        similarity_threshold = 0.7  # Tune this based on testing
        if max(similarities, default=0) < similarity_threshold:
            return "Sorry, I can only answer questions related to the video content. Please ask something about the video."

        return answer
    except Exception as e:
        print(f"❌ RAG error: {e}")
        return "Sorry, I couldn't process your request. Please try again."

# Visual Learning (Mind Map) - Updated Code
def process_visual_learning(cleaned_text):
    print("✅ Starting mind map extraction")
    mindmap_text = extract_mindmap(cleaned_text)
    print("✅ Mind map extracted")
    print("✅ Starting mind map image generation")
    image_path = generate_mindmap_image(mindmap_text)
    print("✅ Mind map image generated")
    display(Image(filename=image_path))
    print("✅ Mind map has been created and saved locally!")

def build_mindmap_prompt(text):
    prompt = f"""
You are an expert in summarization and mind mapping.

Here is a cleaned transcription of a YouTube educational video:

{text}

Your task:
1. Identify the overall MAIN TOPIC (e.g., Machine Learning).
2. Under it, extract the **main branches** (e.g., Supervised Learning, Unsupervised Learning).
3. For each branch, list the **sub-ideas** and details underneath it.

⚠️ IMPORTANT: Follow this exact hierarchy:
- [MAIN TOPIC]
    - Main Branch 1
        - Sub-idea 1
        - Sub-idea 2
    - Main Branch 2
        - Sub-idea 1
        - Sub-idea 2

Be concise but cover all key concepts.

⚠️ Output ONLY the structured list. No introduction, no explanation, no comments.
"""
    return prompt

def extract_mindmap(text):
    prompt = build_mindmap_prompt(text)
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in summarization."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

def parse_outline_to_edges(outline_text):
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
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        lightness = 0.85
        saturation = 0.4
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
        )
        colors.append(hex_color)
    return colors

def generate_mindmap_image(outline_text, output_filename="mindmap_generated"):
    dot = graphviz.Digraph(comment="Mind Map", engine="twopi")
    dot.graph_attr.update(
        splines="curved",
        nodesep="1.0",
        ranksep="2.0",
        overlap="false"
    )
    edges, root = parse_outline_to_edges(outline_text)
    node_hierarchy = {}
    for parent, child in edges:
        if parent not in node_hierarchy:
            node_hierarchy[parent] = []
        node_hierarchy[parent].append(child)
    color_map = {root: "#F7E1A1"}
    top_level_nodes = node_hierarchy.get(root, [])
    pastel_colors = generate_distinct_pastel_colors(len(top_level_nodes))
    for node, color in zip(top_level_nodes, pastel_colors):
        color_map[node] = color
    def propagate_colors(node, parent_color):
        if node not in color_map:
            color_map[node] = parent_color
        if node in node_hierarchy:
            for child in node_hierarchy[node]:
                propagate_colors(child, color_map[node])
    for node in top_level_nodes:
        propagate_colors(node, color_map[node])
    for parent, child in edges:
        parent_color = color_map.get(parent, "#D3D3D3")
        child_color = color_map.get(child, parent_color)
        dot.node(parent, shape="box", style="filled,setlinewidth(2)", fillcolor=parent_color, fontcolor="black", fontsize="12", fontname="Arial")
        dot.node(child, shape="box", style="filled,setlinewidth(2)", fillcolor=child_color, fontcolor="black", fontsize="12", fontname="Arial")
        dot.edge(parent, child, color=parent_color)
    dot.render(output_filename, format='png', cleanup=True)
    print(f"✅ Mind map saved as {output_filename}.png")
    return output_filename + ".png"

# Kinesthetic Learning (Quiz)
def process_kinesthetic_learning(cleaned_text):
    quiz_text = generate_quiz(cleaned_text)
    run_interactive_quiz(quiz_text)

def build_quiz_prompt(text):
    return f"""
You are an expert educator.
Based on the transcript below, generate exactly 5 multiple-choice quiz questions:
Transcript:
{text[:2000]}...
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

def generate_quiz(text):
    prompt = build_quiz_prompt(text)
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a quiz creator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

def run_interactive_quiz(quiz_text):
    print("\n🧠 Starting interactive quiz!")
    questions = quiz_text.strip().split('\n\n')
    score = 0
    for i, q in enumerate(questions):
        lines = q.strip().split('\n')
        question = lines[0]
        choices = lines[1:5]
        correct_line = [l for l in lines if l.lower().startswith("correct")]
        if not correct_line:
            print("⚠️ Skipping malformed question.")
            continue
        correct_answer = correct_line[0].split(":")[1].strip().upper()
        print(f"\nQ{i+1}: {question}")
        for choice in choices:
            print(choice)
        user_answer = input("👉 Your answer (A/B/C/D): ").strip().upper()
        if user_answer == correct_answer:
            print("✅ Correct!")
            score += 1
        else:
            print(f"❌ Wrong. Correct answer was: {correct_answer}")
    print(f"\n🎯 Quiz Complete! Your Score: {score}/{len(questions)}")

# Auditory Learning
def process_auditory_learning(rag_agent, config, transcript_text):
    print("\n🎧 Auditory Learning Mode ON!")
    while True:
        if os.path.exists("user_input.wav"):
            os.remove("user_input.wav")
        if os.path.exists("reply.mp3"):
            os.remove("reply.mp3")
        audio_path = record_audio()
        if not audio_path:
            retry = input("\n🔁 Try recording again? (yes/no): ").strip().lower()
            if retry != 'yes':
                break
            continue
        user_text = transcribe_user_audio(audio_path)
        if not user_text:
            retry = input("\n🔁 Try recording again? (yes/no): ").strip().lower()
            if retry != 'yes':
                break
            continue
        print(f"\n🗣️ You said: {user_text}")
        assistant_text = get_assistant_response(user_text, rag_agent, config, transcript_text)
        print(f"\n🤖 AI said: {assistant_text}")
        audio_response_path = speak_response(assistant_text)
        if audio_response_path:
            display(Audio(audio_response_path))
        again = input("\n🔁 Another question? (yes/no): ").strip().lower()
        if again != 'yes':
            break
    print("🎧 Auditory Learning Mode ended.")

def record_audio(filename="user_input.wav", duration=10):
    print(f"🎤 Recording audio for {duration} seconds...")
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        frames = []
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"✅ Audio recorded: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return None

def transcribe_user_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="en"
            )
        print(f"✅ Transcribed audio: {audio_path}")
        return transcript.text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

def speak_response(text, filename="reply.mp3"):
    try:
        speech = openai.audio.speech.create(model="tts-1", voice="nova", input=text)
        with open(filename, "wb") as f:
            f.write(speech.content)
        print(f"✅ Response saved as: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Speech generation error: {e}")
        return None

# Reading Learning
def process_reading_learning(rag_agent, config, transcript_text):
    print("✅ Bot ready! Ask questions about the video.")
    while True:
        user_input = input("📝 Your question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        response = get_assistant_response(user_input, rag_agent, config, transcript_text)
        print(f"\n🤖 Response: {response}")

# Helper function for smart loop conversation
def suggest_learning_style(current_style):
    """Suggest a different learning style based on the current one."""
    current_style = current_style.lower()
    print(f"DEBUG: Current style: {current_style}")
    styles = {
        'visual': "Would you like to try a mind map to visualize the video content? It’s great for seeing the big picture!",
        'reading': "How about exploring the video with interactive Q&A? You can ask questions and get detailed answers!",
        'kinesthetic': "Why not test your knowledge with a quiz? It’s a fun way to engage with the material!",
        'auditory': "Want to try voice-based learning? You can ask questions and hear the answers!"
    }
    recommendations = {
        "kinesthetic": "auditory",
        "auditory": "kinesthetic",
        "reading": "visual",
        "visual": "reading"
    }
    suggested_style = recommendations.get(current_style, 'reading')
    print(f"DEBUG: Suggested style: {suggested_style}")
    return styles[suggested_style], suggested_style

# Main execution with smart loop
def welcome_message():
    print("🎓 Welcome to SmarterLearn!")
    print("Learning styles: visual (mind maps), reading (Q&A), kinesthetic (quizzes), auditory (voice)")
    print("Enter 'visual', 'reading', 'kinesthetic', or 'auditory'.")

def get_learning_style():
    style = input("👉 Learning style: ").strip().lower()
    if style not in ['visual', 'reading', 'kinesthetic', 'auditory']:
        print("❌ Invalid choice. Please enter 'visual', 'reading', 'kinesthetic', or 'auditory'.")
        return get_learning_style()
    return style

def get_youtube_url():
    return input("🎥 YouTube video URL: ").strip()

if __name__ == "__main__":
    welcome_message()
    while True:
        user_style = get_learning_style()
        youtube_url = get_youtube_url()
        print("⏳ Preparing content...")
        cleaned_text, audio_path = None, None
        while True:  # Loop to handle invalid URLs
            try:
                cleaned_text, audio_path = preprocess_youtube_video(youtube_url)
                break  # Exit URL loop if preprocessing succeeds
            except Exception as e:
                if "Unsupported URL" in str(e):
                    print("❌ Invalid URL: Please enter a valid YouTube URL (e.g., https://youtu.be/...).")
                    youtube_url = get_youtube_url()
                    print("⏳ Preparing content...")
                else:
                    print(f"❌ Error: {e}")
                    print("ℹ️ Check API keys, FFmpeg, Graphviz, network, or microphone.")
                    retry = input("\n🔁 Would you like to retry the same style with a new URL? (yes/no): ").strip().lower()
                    if retry == 'yes':
                        youtube_url = get_youtube_url()
                        print("⏳ Preparing content...")
                    else:
                        break  # Exit URL loop to proceed to style suggestion
        if cleaned_text is None:  # No valid preprocessing, skip to style suggestion
            try_another_style = input("\n🔄 Would you like to try another learning style? (yes/no): ").strip().lower()
            if try_another_style == 'yes':
                continue  # Restart outer loop to choose new style
            break  # Exit to end program
        # Process the selected style
        try:
            if user_style in ["reading", "auditory"]:
                rag_agent, config = setup_rag(cleaned_text)
                if user_style == "reading":
                    process_reading_learning(rag_agent, config, cleaned_text)
                else:
                    process_auditory_learning(rag_agent, config, cleaned_text)
            elif user_style == "visual":
                process_visual_learning(cleaned_text)
            elif user_style == "kinesthetic":
                process_kinesthetic_learning(cleaned_text)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            print("✅ Session completed!")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("ℹ️ Check API keys, FFmpeg, Graphviz, network, or microphone.")
            retry = input("\n🔁 Would you like to retry the same style with a new URL? (yes/no): ").strip().lower()
            if retry == 'yes':
                youtube_url = get_youtube_url()
                print("⏳ Preparing content...")
                cleaned_text, audio_path = None, None
                continue
        # Smart loop: Suggest another learning style
        try:
            suggestion, suggested_style = suggest_learning_style(user_style)
            print(f"\n🌟 {suggestion}")
            try_another = input(f"👉 Try the {suggested_style} style with the same video? (yes/no): ").strip().lower()
            if try_another == 'yes':
                user_style = suggested_style
            else:
                try_another_style = input("\n🔄 Would you like to try another learning style? (yes/no): ").strip().lower()
                if try_another_style == 'yes':
                    new_style = get_learning_style()
                    while new_style == user_style:
                        print(f"❌ Please choose a different style than {user_style}.")
                        new_style = get_learning_style()
                    user_style = new_style
                else:
                    continue_session = input("\n🔄 Would you like to start a new session with a different video? (yes/no): ").strip().lower()
                    if continue_session != 'yes':
                        break
                    continue
            # Ask if the user wants to reuse the current URL or enter a new one
            reuse_url = input("\n🔄 Would you like to use the same video URL? (yes/no): ").strip().lower()
            if reuse_url == 'yes':
                if not cleaned_text:
                    print("❌ No preprocessed video available. Please enter a new URL.")
                    youtube_url = get_youtube_url()
                    cleaned_text, audio_path = preprocess_youtube_video(youtube_url)
            else:
                youtube_url = input("👉 Enter your new YouTube URL: ").strip()
                cleaned_text, audio_path = preprocess_youtube_video(youtube_url)
            # Process the selected style
            try:
                if user_style in ["reading", "auditory"]:
                    rag_agent, config = setup_rag(cleaned_text)
                    if user_style == "reading":
                        process_reading_learning(rag_agent, config, cleaned_text)
                    else:
                        process_auditory_learning(rag_agent, config, cleaned_text)
                elif user_style == "visual":
                    process_visual_learning(cleaned_text)
                elif user_style == "kinesthetic":
                    process_kinesthetic_learning(cleaned_text)
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
                print("✅ Session completed!")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("ℹ️ Check API keys, FFmpeg, Graphviz, network, or microphone.")
        except Exception as e:
            print(f"❌ Error in style suggestion: {e}")
            print("ℹ️ Defaulting to reading style suggestion")
            suggestion, suggested_style = "How about exploring the video with interactive Q&A? You can ask questions and get detailed answers!", "reading"
            try_another = input(f"👉 Try the {suggested_style} style with the same video? (yes/no): ").strip().lower()
            if try_another == 'yes':
                user_style = suggested_style
                try:
                    rag_agent, config = setup_rag(cleaned_text)
                    process_reading_learning(rag_agent, config, cleaned_text)
                    print("✅ Session completed!")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print("ℹ️ Check API keys, FFmpeg, Graphviz, network, or microphone.")
            else:
                try_another_style = input("\n🔄 Would you like to try another learning style? (yes/no): ").strip().lower()
                if try_another_style == 'yes':
                    new_style = get_learning_style()
                    while new_style == user_style:
                        print(f"❌ Please choose a different style than {user_style}.")
                        new_style = get_learning_style()
                    user_style = new_style
                    reuse_url = input("\n🔄 Would you like to use the same video URL? (yes/no): ").strip().lower()
                    if reuse_url == 'yes':
                        if not cleaned_text:
                            print("❌ No preprocessed video available. Please enter a new URL.")
                            youtube_url = get_youtube_url()
                            cleaned_text, audio_path = preprocess_youtube_video(youtube_url)
                    else:
                        youtube_url = input("👉 Enter your new YouTube URL: ").strip()
                        cleaned_text, audio_path = preprocess_youtube_video(youtube_url)
                    try:
                        if user_style in ["reading", "auditory"]:
                            rag_agent, config = setup_rag(cleaned_text)
                            if user_style == "reading":
                                process_reading_learning(rag_agent, config, cleaned_text)
                            else:
                                process_auditory_learning(rag_agent, config, cleaned_text)
                        elif user_style == "visual":
                            process_visual_learning(cleaned_text)
                        elif user_style == "kinesthetic":
                            process_kinesthetic_learning(cleaned_text)
                        if audio_path and os.path.exists(audio_path):
                            os.remove(audio_path)
                        print("✅ Session completed!")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        print("ℹ️ Check API keys, FFmpeg, Graphviz, network, or microphone.")
        continue_session = input("\n🔄 Would you like to start a new session with a different video? (yes/no): ").strip().lower()
        if continue_session != 'yes':
            break
    print("🎉 Thank you for using SmarterLearn!")