# 🕷️ AI Crawler – Web Page Question Answering Chatbot

This app allows users to enter a URL and ask questions about the content of that page. It uses Selenium to scrape the website, splits the text into chunks, stores it in a vector database, and retrieves relevant information to answer user queries using a local language model.

---

## ⚙️ How It Works

1. User enters a URL.
2. The content of the page is scraped using Selenium.
3. Text is split into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.
4. Chunks are embedded and stored in an in-memory vector store.
5. User asks a question in the chat.
6. The app retrieves similar chunks from the vector store and sends them to the LLM for answering.

---

## 🛠️ Setup Instructions

1. Clone this repository. (This project is built using Python 3.10.16)
2. Install all dependencies and run the app using the following commands:

```bash
pip install -r requirements.txt
ollama run llama3.2
streamlit run ai_scraper.py
