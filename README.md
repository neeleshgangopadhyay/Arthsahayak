# Arthsahayak
An Agentic RAG project for understanding Indian Economics — question answering over RBI reports, Economic Surveys, NCERT textbooks, and budget documents.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies: 
use the command "python -m pip install -r requirements.txt"
3.  Copy `.env.example` to `.env` and fill in your own Groq API key:
GROQ_API_KEY=your_key_here
4. Build the vector index from the PDFs in `data/`: 
python src/ingest.py
5. Try the agent: 
python src/agents.py



Full architecture notes, evaluation results, and usage details are being written up as the project nears completion (see `ArthSahayak - Project Log.md` for current progress).
