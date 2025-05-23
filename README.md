# SmarterLearn Project

Welcome to **SmarterLearn**, an innovative learning assistant designed to enhance educational experiences by adapting to individual learning styles. This project includes two main components: a Streamlit-based web application (`SmarterLearnApp.py`) and a standalone Python script (`LearningTools.py`) without a graphical interface. Both leverage AI to process YouTube educational videos, offering personalized learning through visual, auditory, kinesthetic, and reading/writing styles.

## Project Overview

- **Purpose**: SmarterLearn helps users learn from YouTube videos by providing tailored content based on their preferred learning style. It transcribes videos, generates mind maps, quizzes, and answers questions using Retrieval-Augmented Generation (RAG) with AI models.
- **Streamlit App (`SmarterLearnApp.py`)**: A web-based interface where users can input a YouTube URL, select a learning style, and interact with the content interactively.
- **Standalone Script (`LearningTools.py`)**: A non-Streamlit version for command-line or script-based usage, offering the same core functionality without a UI.
- **Learning Styles Supported**: Visual (mind maps), Auditory (voice questions and responses), Kinesthetic (quizzes), and Reading/Writing (Q&A).

## Features

- **Video Processing**: Downloads and transcribes YouTube videos using `yt-dlp` and OpenAI's Whisper.
- **Learning Style Customization**: Generates mind maps, quizzes, or conversational responses based on user preferences.
- **AI-Powered**: Utilizes OpenAI's GPT-4o-mini for summarization, question answering, and quiz generation, with LangChain for RAG.
- **Caching**: Implements caching to improve performance for repeated video processing.
- **User-Friendly**: Streamlit app provides an intuitive interface with style suggestions and history tracking.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Git**: For cloning the repository.
- **Internet Connection**: Required for downloading videos and API calls.

## Installation

### Clone the Repository
```bash
git clone https://github.com/RaghadAlmutairi/SmarterLearn.git
cd SmarterLearn
```

### Set Up a Virtual Environment
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

### Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Configure API Keys
- Obtain API keys for OpenAI and set them as environment variables:
  - Create a `.env` file in the project root:
    ```
    OPENAI_API_KEY=your_openai_api_key
    ```
  - Install `python-dotenv` (included in `requirements.txt`) to load the keys automatically.

## Usage

### Running the Streamlit App (`SmarterLearnApp.py`)
1. Ensure the virtual environment is activated.
2. Start the Streamlit app:
   ```bash
   streamlit run SmarterLearnApp.py
   ```
3. Open your browser and go to the provided local URL (e.g., `http://localhost:8501`).
4. Paste a YouTube video URL, choose a learning style, and follow the interactive prompts.

### Running the Standalone Script (`LearningTools.py`)
1. Ensure the virtual environment is activated.
2. Run the script with a YouTube URL as an argument:
   ```bash
   python LearningTools.py https://www.youtube.com/watch?v=example
   ```
3. Follow the command-line prompts to select a learning style or process the video manually.

## File Structure
- `.gitignore`: Excludes the virtual environment and other unnecessary files from Git.
- `LearningTools.py`: Core logic without Streamlit, suitable for script-based usage.
- `README.md`: This file, providing project documentation.
- `SmarterLearnApp.py`: The Streamlit-based web application.
- `requirements.txt`: List of Python dependencies.

## Dependencies
The project relies on the following packages (detailed in `requirements.txt`):
- `streamlit`: For the web interface.
- `yt-dlp`: For downloading YouTube videos.
- `pydub`: For audio processing.
- `langchain-openai`, `langchain-huggingface`, `langchain-chroma`: For RAG and embeddings.
- `openai`: For AI model access (Whisper and GPT).
- `graphviz`: For generating mind maps.
- `pillow`, `numpy`, `pandas`, `matplotlib`: For visualization and data handling.

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Describe your changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request on GitHub.

## Issues and Support
- Report bugs or suggest features by opening an issue on the [GitHub Issues page](https://github.com/RaghadAlmutairi/SmarterLearn/issues).
- For questions, contact the maintainer via GitHub.

## Acknowledgments
- **SDA and Ironhack**: For providing the AI Engineer Bootcamp, where I gained the skills to build this project.
- **Open-Source Communities**: Thanks to Streamlit, LangChain, and OpenAI for their tools.
- **Instructors and Peers**: For guidance and support throughout the bootcamp.

---

*Last updated: May 06, 2025*
