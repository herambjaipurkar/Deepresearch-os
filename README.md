# DeepResearch-OS 🌐

An autonomous, stateful multi-agent research workspace that scrapes the web, synthesizes deep technical intelligence reports, and subjects its own outputs to human-grade quality assurance audits. Built using **LangGraph**, **Gemini 1.5 Flash**, and **Llama-3 (Groq)**.

## 🚀 Key Architectural Features: 

- **Cyclic Agent Logic Loops:** Leverages LangGraph to build multi-agent states where a **Critic Agent** programmatically approves or rejects reports written by a **Writer Agent**, sending revisions back to a **Web Surfer Tool**.
- 
- **Stateful Memory Infrastructure:** Uses native memory checkpointers to preserve full trace histories across operational loops.
- 
- **Cost-Optimized Hybrid Topology:** Pairs ultra-low latency inference models (Groq) for rapid agent routing logic alongside large reasoning windows (Gemini API) for processing massive document summaries completely on **100% Free Tiers**.

## 🛠️ Tech Stack
- **Orchestration Framework:** LangGraph
- **LLM Engine Components:** Gemini-2.5-Flash & Llama3-70b (via Groq)
- **External Interfaces:** DuckDuckGo Web Scraper API
- **UI Architecture Dashboard:** Streamlit Community Engine


## 📦 Local Installation Guide

1. Clone the project and navigate to the directory:
```bash
git clone [https://github.com/YOUR_USERNAME/deepresearch-os.git](https://github.com/YOUR_USERNAME/deepresearch-os.git)
cd deepresearch-os
```

2. Establish and source an insulated Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install production-ready dependency versions:
```bash
pip install -r requirements.txt
```

4. Boot up the Streamlit interface dashboard:
```bash
streamlit run app.py
```
