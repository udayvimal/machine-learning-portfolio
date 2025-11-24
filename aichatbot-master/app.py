# medibot.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from custom_mistral_llm import MistralChatLLM

DB_FAISS_PATH = "vectorstore/db_faiss"

PROMPT_TEMPLATE = """
You are a helpful medical assistant.

Your job is to answer the user's medical question using relevant medical symptoms and records extracted from the documents (if any). 

❗ Do NOT show raw context, case numbers, or JSON-like structure.  
❗ ONLY show:
- A short answer mentioning up to 2–3 possible diseases.
- Brief reasoning based on symptoms.
- (Optional) A recommendation like "consult a doctor".

If not enough information is found, just say: “I’m not sure based on the current data.”

[The following information is **only for your internal use** — do NOT include it in your answer.]
Context: {context}

Question: {question}

Answer:
"""




def set_custom_prompt(template: str):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def build_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    return RetrievalQA.from_chain_type(
        llm=MistralChatLLM(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(PROMPT_TEMPLATE)},
        input_key="question",
    )
