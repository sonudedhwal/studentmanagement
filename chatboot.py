import os
import re
import joblib
import numpy as np
import streamlit as st
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Bank Chatbot", page_icon="💬")

STATE_FILE = "bank_state.joblib"      # saved by your main bank app (same folder)
CHAT_MIN_CONF = 0.45
CHAT_LLM_MODEL = "claude-sonnet-5-5"


# ---------- load the bank data (read-only) ----------
def load_bank():
    if not os.path.exists(STATE_FILE):
        return {}, []
    try:
        state = joblib.load(STATE_FILE)
        return state.get("accounts", {}), state.get("flagged", [])
    except Exception:
        return {}, []


# ---------- NLP intent model (TF-IDF + Logistic Regression) ----------
INTENT_EXAMPLES = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening", "namaste"],
    "help": ["help", "what can you do", "how do I use this", "show commands"],
    "check_balance": ["what is the balance of account 1001", "check balance 1001",
                      "how much money is in account 1001", "balance please"],
    "history": ["show history of account 1001", "last transactions for 1001",
                "transaction history", "recent activity"],
    "bank_summary": ["how many accounts are there", "total money in the bank",
                     "bank summary", "how much money does the bank hold"],
    "fraud_status": ["any fraud", "were any transactions flagged",
                     "suspicious transactions", "show unusual activity"],
    "money_action": ["deposit 500 into account 1001", "withdraw 200 from account 1001",
                     "transfer 300 from account 1001 to account 1002",
                     "send 50 from 1001 to 1002"],
}


@st.cache_resource
def get_intent_model():
    X, y = [], []
    for intent, phrases in INTENT_EXAMPLES.items():
        for p in phrases:
            X.append(p)
            y.append(intent)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                          LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model


def predict_intent(text):
    model = get_intent_model()
    probs = model.predict_proba([text])[0]
    i = int(np.argmax(probs))
    return model.classes_[i], float(probs[i])


# ---------- Claude (LLM) fallback ----------
def ask_llm(question, accounts, flagged):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "I am not sure about that. Set the ANTHROPIC_API_KEY environment variable so the AI can answer."
    total = sum(a["balance"] for a in accounts.values())
    context = (f"The bank has {len(accounts)} accounts holding {total:,.2f} in total, "
               f"and {len(flagged)} transactions were flagged by the fraud model.")
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_log[-10:]]
    while history and history[0]["role"] != "user":
        history.pop(0)
    history.append({"role": "user", "content": question})
    try:
        reply = anthropic.Anthropic().messages.create(
            model=CHAT_LLM_MODEL, max_tokens=600,
            system=("You are a friendly assistant for a bank management app used by people "
                    "who may be blind, deaf or less literate. Use short, simple sentences. "
                    "Never ask for or reveal PINs or OTPs. You cannot move money. " + context),
            messages=history,
        )
        return reply.content[0].text
    except Exception as e:
        return f"Sorry, the AI service could not be reached: {e}"


# ---------- the chatbot brain ----------
def chatbot_reply(text, accounts, flagged):
    intent, conf = predict_intent(text)
    if conf < CHAT_MIN_CONF:
        return ask_llm(text, accounts, flagged), "🤖 Claude (LLM)"
    src = f"🧠 ML intent: {intent} ({conf:.0%})"

    nums = [n for n in re.findall(r"\d+", text) if n in accounts]

    if intent == "greeting":
        return "Hello! Ask me for a balance, history, or a bank summary.", src
    if intent == "help":
        return "I can check a balance, show history, give a bank summary, show fraud flags, and answer general questions.", src
    if intent == "bank_summary":
        total = sum(a["balance"] for a in accounts.values())
        return f"There are {len(accounts)} accounts holding {total:,.2f} in total.", src
    if intent == "fraud_status":
        return f"The AI fraud model has flagged {len(flagged)} unusual transactions.", src
    if intent == "money_action":
        return "For safety, deposits, withdrawals and transfers must be done in the main bank app.", src

    # balance / history need an account number
    if not nums:
        return "Which account? Please say the account number.", src
    n = nums[0]
    if accounts[n].get("pin_hash"):
        return f"Account {n} is PIN-protected, so I cannot show it here. Use the main bank app.", src
    if intent == "check_balance":
        return f"Account {n} belongs to {accounts[n]['name']} and has {accounts[n]['balance']:,.2f}.", src
    hist = accounts[n].get("history", [])[-5:]
    return ("Latest activity:\n\n" + "\n\n".join(reversed(hist))) if hist else "No transactions yet.", src


# ---------- page ----------
st.title("💬 Bank Chatbot")
st.caption("ML intent model first, Claude LLM for everything else.")

accounts, flagged = load_bank()
if not accounts:
    st.info(f"No saved bank data found ('{STATE_FILE}'). Run your main bank app first, "
            "or put this file in the same folder as bank_state.joblib.")

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

for m in st.session_state.chat_log:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("src"):
            st.caption(m["src"])

question = st.chat_input("Ask something, e.g. 'balance of account 1001'")
if question:
    st.session_state.chat_log.append({"role": "user", "content": question})
    answer, src = chatbot_reply(question, accounts, flagged)
    st.session_state.chat_log.append({"role": "assistant", "content": answer, "src": src})
    st.rerun()