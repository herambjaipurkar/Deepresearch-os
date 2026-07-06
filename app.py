import streamlit as st
import os
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field

# LangChain & LangGraph Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. DEFINE SYSTEM STATE
class ResearchState(TypedDict):
    research_topic: str
    research_notes: List[str]
    report: str
    critique: str
    loop_count: int
    messages: List[BaseMessage]

# 2. INITIALIZE TOOLS & MODELS
search_tool = DuckDuckGoSearchRun()

def get_models(gemini_key, groq_key):
    """Initializes advanced models using the user's free API keys"""
    # UPDATE: We upgraded this to gemini-2.5-flash as 1.5 was recently deprecated by Google
    brain_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=gemini_key,
        temperature=0.3
    )
    # UPDATE: We upgraded this to llama-3.3-70b-versatile as the older llama3 was decommissioned by Groq
    router_llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=groq_key,
        temperature=0.1
    )
    return brain_llm, router_llm

# 3. DEFINE AGENT NODES
def researcher_node(state: ResearchState):
    """Autonomous Surfer Agent: Gathers live intelligence from the web"""
    st.toast("🕵️‍♂️ Researcher Agent is searching the web...", icon="🔍")
    topic = state["research_topic"]
    loop_count = state.get("loop_count", 0)
    
    # Adapt search query if the critic provided feedback
    search_query = topic
    if state.get("critique"):
        search_query = f"{topic} focus on: {state['critique'][:100]}"
        
    try:
        raw_results = search_tool.run(search_query)
    except Exception:
        raw_results = f"No additional search listings found for query: {search_query}"
        
    notes = state.get("research_notes", [])
    notes.append(f"[Round {loop_count}] {raw_results}")
    
    return {
        "research_notes": notes,
        "loop_count": loop_count + 1
    }

def writer_node(state: ResearchState, brain_llm):
    """Synthesizer Agent: Compiles scattered raw text into a professional report"""
    st.toast("✍️ Writer Agent is synthesizing report...", icon="📝")
    topic = state["research_topic"]
    notes = "\n\n".join(state["research_notes"])
    critique = state.get("critique", "None")
    
    prompt = f"""You are an Expert Technical Writer. Your task is to write a comprehensive, highly-detailed, markdown-formatted report on the topic: '{topic}'.
    
    Use the following raw research notes gathered by your web surfer:
    ---
    {notes}
    ---
    
    Previous Critic Review Comments (if any): {critique}
    
    Requirements:
    1. Organize with clear headers, bold text, and bullet points.
    2. Include an Executive Summary, Detailed Analysis, Pros/Cons, and Future Outlook.
    3. Cite data point findings cleanly. Never make up facts.
    """
    
    response = brain_llm.invoke([HumanMessage(content=prompt)])
    return {"report": response.content}

def critic_node(state: ResearchState, router_llm):
    """The Quality Gatekeeper: Evaluates reports to kill hallucinations"""
    st.toast("⚖️ Critic Agent is auditing report quality...", icon="🛡️")
    topic = state["research_topic"]
    report = state["report"]
    
    prompt = f"""You are a Strict Quality Control Auditor. Review this drafted report for the topic '{topic}'.
    Determine if the report is thoroughly comprehensive or if it lacks depth/contains gaps.
    
    Report:
    {report}
    
    Respond in EXACTLY the following structure. Do not include any other text:
    CRITIQUE: [Provide constructive feedback or write 'PASSED' if the report is perfect and completely covers the topic]
    """
    
    response = router_llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    critique_val = "PASSED"
    if "CRITIQUE:" in content:
        critique_val = content.split("CRITIQUE:")[1].strip()
        
    return {"critique": critique_val}

# 4. ROUTING LOGIC
def route_approval(state: ResearchState):
    """Evaluates if the system should terminate or continue gathering data"""
    critique = state.get("critique", "")
    loop_count = state.get("loop_count", 0)
    
    # Prevent infinite loops on free tiers by hard capping at 3 loops
    if "PASSED" in critique.upper() or loop_count >= 3:
        st.toast("🎉 Quality checks passed! Finalizing...", icon="✅")
        return END
    else:
        st.toast(f"🔄 Report rejected by Critic. Routing back to Researcher (Iteration {loop_count})", icon="⚠️")
        return "researcher"

# 5. GRAPH BUILDER ENGINE
def build_research_graph(brain_llm, router_llm):
    workflow = StateGraph(ResearchState)
    
    # Add Nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", lambda state: writer_node(state, brain_llm))
    workflow.add_node("critic", lambda state: critic_node(state, router_llm))
    
    # Setup Flow Dependencies
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "critic")
    
    # Add Conditional Routing Edge
    workflow.add_conditional_edges(
        "critic",
        route_approval,
        {
            "researcher": "researcher",
            END: END
        }
    )
    
    return workflow.compile(checkpointer=MemorySaver())

# 6. STREAMLIT FRONTEND USER INTERFACE
st.set_page_config(page_title="DeepResearch-OS", page_icon="🤖", layout="wide")

st.title("🌐 DeepResearch-OS")
st.caption("Autonomous Stateful Multi-Agent Research System Powered by LangGraph")

# Sidebar Configuration for API Credentials
with st.sidebar:
    st.header("🔑 API Authentication")
    st.markdown("Get keys for free at [Google AI Studio](https://aistudio.google.com/) & [Groq Console](https://console.groq.com/)")
    gemini_key = st.text_input("Google Gemini API Key", type="password")
    groq_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    st.markdown("""
    ### How it works:
    1. **Surfer Agent** queries the web via DuckDuckGo.
    2. **Writer Agent** builds a deep markdown report via Gemini.
    3. **Critic Agent** audits findings via Llama3.
    4. System loops autonomously until quality passes.
    """)

# Main Content Area
query = st.text_input("What complex topic or market trend do you want investigated?", 
                      placeholder="e.g., Comparative analysis of modern vector databases in 2026")

if st.button("Launch Autonomous Research Run", type="primary"):
    if not gemini_key or not groq_key:
        st.error("Please provide both your Gemini and Groq API keys in the sidebar to run the system.")
    elif not query.strip():
        st.warning("Please type a valid research topic.")
    else:
        with st.spinner("Agents are collaborating... Please watch toasts below for operational logs."):
            # Initialize Models & Compile Graph Pipeline
            brain_llm, router_llm = get_models(gemini_key, groq_key)
            app = build_research_graph(brain_llm, router_llm)
            
            # Execute Pipeline State Graph
            initial_state = {
                "research_topic": query,
                "research_notes": [],
                "report": "",
                "critique": "",
                "loop_count": 0,
                "messages": []
            }
            
            config = {"configurable": {"thread_id": "session_" + str(hash(query))}}
            final_output = app.invoke(initial_state, config=config)
            
            # Display Generated Results
            st.success("🤖 Research Complete!")
            
            tab1, tab2 = st.tabs(["📄 Compiled Research Report", "⚙️ Meta-Agent Insights"])
            
            with tab1:
                st.markdown(final_output.get("report", "No report generated."))
                
            with tab2:
                st.subheader("System Execution Log Details")
                st.markdown(f"**Total Loop Iterations:** {final_output.get('loop_count')}")
                st.markdown(f"**Final Critic Review Audit:** {final_output.get('critique')}")
                with st.expander("See Raw Scraped Research Notes Context"):
                    st.write(final_output.get("research_notes"))