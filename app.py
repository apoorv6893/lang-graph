import streamlit as st
from typing import TypedDict, Optional
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# -------------------------------
# PAGE
# -------------------------------
st.set_page_config(page_title="Loan Approval - LangGraph", layout="wide")
st.title("🏦 Loan Approval System (LangGraph Demo)")

# -------------------------------
# LLM
# -------------------------------
def get_llm(api_key, model, temp):
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model,
        temperature=temp
    )

# -------------------------------
# STATE
# -------------------------------
class LoanState(TypedDict):
    name: str
    income: float
    credit_score: int
    loan_amount: float
    risk: Optional[str]
    reason: Optional[str]
    decision: Optional[str]
    human_notes: Optional[str]
    income_verified: Optional[str]
    iteration: int


# -------------------------------
# NODE: AI RISK ASSESSMENT
# -------------------------------
def risk_assessment_node(state: LoanState):
    llm = st.session_state.llm

    prompt = f"""
You are a loan risk evaluator.

Customer:
Name: {state['name']}
Income: {state['income']}
Credit Score: {state['credit_score']}
Loan Amount: {state['loan_amount']}

Human Inputs:
Notes: {state.get('human_notes')}
Income Verified: {state.get('income_verified')}

Classify risk as:
- low
- medium
- high

Return STRICT JSON:
{{"risk": "...", "reason": "..."}}
"""

    response = llm.invoke(prompt).content

    # ✅ Proper JSON parsing
    try:
        parsed = json.loads(response)
        risk = parsed.get("risk", "medium").lower()
        reason = parsed.get("reason", response)
    except:
        risk = "medium"
        reason = response

    return {
        "risk": risk,
        "reason": reason,
        "iteration": state["iteration"] + 1
    }


# -------------------------------
# ROUTING
# -------------------------------
def route(state: LoanState):
    if state["risk"] == "low":
        return "approve"
    elif state["risk"] == "high":
        return "reject"
    else:
        return "human"


# -------------------------------
# FINAL NODES
# -------------------------------
def approve_node(state):
    return {"decision": "APPROVED"}


def reject_node(state):
    return {"decision": "REJECTED"}


# -------------------------------
# GRAPH
# -------------------------------
def build_graph():
    builder = StateGraph(LoanState)

    builder.add_node("risk_assessment", risk_assessment_node)
    builder.add_node("approve", approve_node)
    builder.add_node("reject", reject_node)

    builder.set_entry_point("risk_assessment")

    builder.add_conditional_edges(
        "risk_assessment",
        route,
        {
            "approve": "approve",
            "reject": "reject",
            "human": END  # pause for human
        }
    )

    builder.add_edge("approve", END)
    builder.add_edge("reject", END)

    return builder.compile()


# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Settings")

api_key = st.sidebar.text_input("Gemini API Key", type="password")
model = st.sidebar.selectbox(
    "Model",
    ["models/gemini-2.5-flash", "models/gemini-1.5-pro"]
)
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

# -------------------------------
# INPUT
# -------------------------------
st.subheader("👤 Applicant Details")

name = st.text_input("Name")
income = st.number_input("Monthly Income", value=50000)
credit = st.number_input("Credit Score", value=650)
loan = st.number_input("Loan Amount", value=200000)

run = st.button("🚀 Evaluate Loan")

# -------------------------------
# INITIAL RUN
# -------------------------------
if run:
    if not api_key:
        st.error("Enter API key")
        st.stop()

    st.session_state.llm = get_llm(api_key, model, temp)
    graph = build_graph()

    state = {
        "name": name,
        "income": income,
        "credit_score": credit,
        "loan_amount": loan,
        "iteration": 0
    }

    result = graph.invoke(state)
    st.session_state.state = result


# -------------------------------
# DISPLAY
# -------------------------------
if "state" in st.session_state:
    s = st.session_state.state

    st.subheader("🤖 AI Assessment")
    st.write(f"Risk: **{s.get('risk')}**")
    st.write(f"Reason: {s.get('reason')}")
    st.write(f"Iteration: {s.get('iteration')}")

    # -------------------------------
    # HUMAN LOOP
    # -------------------------------
    if s.get("risk") == "medium" and s["iteration"] < 3 and not s.get("decision"):

        st.subheader("🧑‍💻 Human Review")

        notes = st.text_area("Reviewer Notes")
        verified = st.selectbox("Income Verified?", ["Yes", "No"])

        col1, col2, col3 = st.columns(3)

        # ✅ APPROVE (state update)
        with col1:
            if st.button("✅ Approve"):
                st.session_state.state["decision"] = "APPROVED"
                st.session_state.state["human_notes"] = notes
                st.rerun()

        # ✅ REJECT (state update)
        with col2:
            if st.button("❌ Reject"):
                st.session_state.state["decision"] = "REJECTED"
                st.session_state.state["human_notes"] = notes
                st.rerun()

        # ✅ LOOP (re-evaluate)
        with col3:
            if st.button("🔁 Re-evaluate"):
                graph = build_graph()

                new_state = {
                    **s,
                    "human_notes": notes,
                    "income_verified": verified
                }

                result = graph.invoke(new_state)
                st.session_state.state = result
                st.rerun()

    # -------------------------------
    # FINAL OUTPUT
    # -------------------------------
    if s.get("decision"):
        st.subheader("🏁 Final Decision")
        st.success(s.get("decision"))

    # -------------------------------
    # DEBUG STATE (POWERFUL FOR DEMO)
    # -------------------------------
    with st.expander("🔍 View Full State"):
        st.json(s)
