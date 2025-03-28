# Aviation Safety & Crash Analysis AI

## Overview

A domain-specific AI designed to help aviation professionals, journalists, and researchers analyze past plane crashes, identify patterns, and suggest safety improvements.

## Features

- Document ingestion from PDFs, TXTs, and URLs
- Vector-based semantic search
- RAG-powered question answering
- Modern Next.js frontend
- FastAPI backend

## Tech Stack

- **Frontend:** Next.js, TypeScript, TailwindCSS
- **Backend:** FastAPI, Python
- **AI/ML:** OpenAI GPT-3.5, LangChain
- **Vector Store:** ChromaDB
- **Document Processing:** PyPDF2, docx2txt

## Prerequisites

- Python 3.8+
- Node.js 16+
- OpenAI API key

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/aviation-safety-ai.git
cd aviation-safety-ai
```

2. Set up the backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. Set up the frontend:

```bash
cd ../frontend
npm install
cp .env.example .env.local
# Edit .env.local with your API URL
```

4. Add your documents:

- Place your aviation safety documents (PDFs, TXTs) in the `backend/data` directory
- Supported formats: PDF, TXT, DOCX

## Running the Application

1. Start the backend server:

```bash
cd backend
uvicorn app:app --reload
```

2. Start the frontend server:

```bash
cd frontend
npm run dev
```

3. Access the application:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Usage

1. The system automatically indexes documents on startup
2. Ask questions about aviation safety through the chat interface
3. The system will search through the indexed documents and provide relevant answers
4. Sources are included with each response

## API Endpoints

- `POST /chat`: Send questions and receive AI-generated responses
- `POST /index`: Manually trigger document re-indexing
- `GET /health`: Health check endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-3.5 API
- LangChain for RAG implementation
- FastAPI for the backend framework
- Next.js for the frontend framework

---

🚀 **Aviation Safety & Crash Analysis AI – Making the skies safer through data-driven insights.**
