# SQL Chatbot Agent

FastAPI service that answers questions about a PostgreSQL database using `google/gemma-4-E4B-it` through Hugging Face Transformers.

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with your PostgreSQL URL and Hugging Face token. Accept the Gemma 4 license on Hugging Face before downloading the model.

Start the API:

```powershell
py -3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Ask a question

```powershell
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Which customers placed the most orders?\", \"debug\": true}"
```

With `"debug": true`, each agent step (model turn and tool result) is written to the application log and returned in `steps`.
