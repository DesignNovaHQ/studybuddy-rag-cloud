# StudyBuddy RAG (cloud version) 📚

Upload a PDF and ask questions in **English, Urdu or Roman Urdu**. Answers cite the file and page,
and the app says "not found" instead of guessing. It can also generate MCQ quizzes from a page range.

## Stack
Streamlit, Groq API (free tier), fastembed multilingual embeddings (ONNX), NumPy cosine search, pypdf.

## Run locally
1. `pip install -r requirements.txt`
2. Get a free key at https://console.groq.com/keys
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your key
4. `streamlit run app.py`

## Deploy free on Streamlit Community Cloud
1. Push this folder to a **public** GitHub repo (the `.gitignore` keeps your key and PDFs out).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings -> Secrets** and paste: `GROQ_API_KEY = "gsk_..."`
5. Click **Deploy**. The first start takes a few minutes while the embedding model downloads.

## Notes
- Visitors upload their own PDFs; nothing is stored on the server (in-memory per session).
- Index one chapter at a time (up to 150 pages per run) to stay inside free memory limits.
- Matching text snippets are sent to the Groq API, so do not upload confidential documents.
- Offline version (Ollama, 100% local) lives in a separate folder/repo.
