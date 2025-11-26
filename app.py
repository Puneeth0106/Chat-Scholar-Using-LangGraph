import os
import re
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

# LangChain / embeddings / vectorstore imports (canonical locations)
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# OpenAI official client (optional) - used only if OPENAI_API_KEY is provided
from openai import OpenAI

# Read API key from environment; fallback to placeholder if not set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "KEY_HERE")

start_greeting = ["hi","hello"]
end_greeting = ["bye"]
way_greeting = ["who are you?"]

#Using this folder for storing the uploaded docs. Creates the folder at runtime if not present
DATA_DIR = "__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

# Initialize OpenAI client only when an API key is available
openai_client = None
if OPENAI_API_KEY and OPENAI_API_KEY != "KEY_HERE":
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

class HumanMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'HumanMessage(content={self.content})'

class AIMessage:
    def __init__(self, content):
        self.content = content
    
    def __repr__(self):
        return f'AIMessage(content={self.content})'


def get_pdf_text(pdf_docs):
    text = ""
    pdf_txt = ""
    for pdf in pdf_docs:
        filename = os.path.join(DATA_DIR,pdf.filename)
        pdf_txt = ""
        # PdfReader accepts a file-like object
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
            pdf_txt += page_text

        with (open(filename, "w", encoding="utf-8")) as op_file:
            op_file.write(pdf_txt)

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # OpenAIEmbeddings will use the OPENAI_API_KEY from the environment
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # We avoid using LangChain's ConversationalRetrievalChain and
    # ConversationBufferMemory. This function is kept for compatibility but
    # returns None. The app uses the manual `answer_question` flow instead.
    return None


def create_context_from_docs(docs, max_chars=2000):
    parts = []
    for d in docs:
        text = getattr(d, "page_content", None) or str(d)
        text = text.replace("\n", " ")[:1000]
        src = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            src = d.metadata.get("source") or d.metadata.get("filename")
        if src:
            parts.append(f"[source: {src}] {text}")
        else:
            parts.append(text)
    joined = "\n\n".join(parts)
    return joined[:max_chars]


def answer_question(question, k=4):
    """Retrieve relevant chunks from the local FAISS vectorstore and call the
    OpenAI chat completion endpoint directly. Returns (answer, docs).
    """
    global vectorstore, chat_history, openai_client

    if vectorstore is None:
        return "Vector store not initialized. Upload PDFs first.", []

    # Try similarity_search; fallback to as_retriever if needed
    try:
        docs = vectorstore.similarity_search(question, k=k)
    except Exception:
        try:
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(question)
        except Exception:
            docs = []

    context = create_context_from_docs(docs)

    system_prompt = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If the context doesn't contain the answer, reply: 'I don't know based on the provided documents.' "
        "When you use content from a document, add a short citation in square brackets using the document's source."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history:
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            messages.append({"role": turn["role"], "content": turn["content"]})
        elif isinstance(turn, HumanMessage):
            messages.append({"role": "user", "content": turn.content})
        elif isinstance(turn, AIMessage):
            messages.append({"role": "assistant", "content": turn.content})

    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    if openai_client is None:
        return "OpenAI client not configured. Set OPENAI_API_KEY.", docs

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0,
        max_tokens=800,
    )

    answer = response.choices[0].message.content

    # persist history
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, docs

def _grade_essay(essay):
    # Use the OpenAI client (if available) to grade the essay. If the client
    # is not initialized, return a helpful error message.
    if openai_client is None:
        return "Error: OpenAI API key not configured. Set OPENAI_API_KEY env var."

    system_prompt = (
        "You are a Chinese bot. Carefully grade the essay based on the given rubric and respond in Chinese only. "
        + rubric_text
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "ESSAY: " + essay}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=1500,
    )

    data = response.choices[0].message.content
    data = re.sub(r'\n', '<br>', data)
    return data


@app.route('/')
def home():
    return render_template('new_home.html')


@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    # We do not create a LangChain conversational chain; use manual flow.
    conversation_chain = None
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history
    msgs = []
    sources = []
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        # Use manual retrieval + LLM answer flow
        if vectorstore is None:
            return redirect('/')

        answer, docs = answer_question(user_question)
        msgs = answer
        sources = []
        for d in docs:
            try:
                md = getattr(d, 'metadata', None) or {}
                if isinstance(md, dict):
                    sources.append(md.get('source') or md.get('filename'))
            except Exception:
                sources.append(None)
    return render_template('new_chat.html', chat_history=chat_history, msgs=msgs, sources=sources)

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('new_pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    result = None
    text = ""
    if request.method == 'POST':
        if request.form.get('essay_rubric', False):
            global rubric_text
            rubric_text = request.form.get('essay_rubric')

            return render_template('new_essay_grading.html')
        
        uploaded = request.files.get('file')
        if uploaded and uploaded.filename:
            pdf_file = uploaded
            text = extract_text_from_pdf(pdf_file)
            result = _grade_essay(text)
        else:
            text = request.form.get('essay_text', '')
            if text:
                result = _grade_essay(text)
            else:
                result = "No essay text provided."
    
    return render_template('new_essay_grading.html', result=result, input_text=text)
    
@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('new_essay_rubric.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

if __name__ == '__main__':
    app.run(debug=True)
