import streamlit as st
from medibot import build_qa_chain

# 🌟 1. Title & Instructions
st.title("🧠 MediBot - Medical Chat Assistant")
st.markdown("Ask any medical question. If PDFs were uploaded previously, the bot will use them for better answers.")

# 🌟 2. Optional PDF Upload
uploaded_files = st.file_uploader("📄 (Optional) Upload new PDF(s)", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("✅ PDFs uploaded. Re-run `create_memory_for_llm.py` to update knowledge base.")

# 🌟 3. Build QA Chain
qa_chain = build_qa_chain()

# 🌟 4. Chat Interface
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("### 💬 Chat with MediBot")
sample_q = st.radio("🧪 Try a sample question:", ["", "What causes iron deficiency anemia?", "Symptoms of liver failure?", "How to manage hypertension?"])
user_query = st.text_input("💬 You:", sample_q, key="user_input")

if user_query:
    # Combine recent turns for minimal context
    contexted_query = ""
    for turn in st.session_state.history[-2:]:
        contexted_query += f"User: {turn['user']}\nAI: {turn['bot']}\n"
    contexted_query += f"User: {user_query}"

    # Get the answer from the QA chain
    result = qa_chain.invoke({"question": contexted_query})
    bot_reply = result["result"].strip()

    # Optional: clean up unwanted preamble
    if "Answer:" in bot_reply:
        bot_reply = bot_reply.split("Answer:", 1)[-1].strip()

    # Display only current exchange
    st.markdown(f"**You:** {user_query}")
    st.markdown(f"**🤖 Bot:** {bot_reply}")

    # Update session history
    st.session_state.history.append({"user": user_query, "bot": bot_reply})

# 🌟 Optional full history
if st.checkbox("📜 Show full chat history"):
    st.markdown("---")
    for chat in st.session_state.history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**🤖 Bot:** {chat['bot']}")
