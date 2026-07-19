# -*- coding: utf-8 -*-
import codecs
import re

html_path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"

with codecs.open(html_path, "r", "utf-8") as f:
    content = f.read()

# 1. Update ChatRequest
old_chat_req = """class ChatRequest(BaseModel):
    message: str
    case_id: str = None"""

new_chat_req = """class ChatRequest(BaseModel):
    message: str
    case_id: str = None
    session_id: str = None

# In-memory chat sessions and KB
import json
chat_sessions = {}

KB_DATA_CACHE = []
try:
    with open(os.path.join(base_dir, "static", "kb_data.json"), "r", encoding="utf-8") as f:
        KB_DATA_CACHE = json.load(f)
except Exception as e:
    logger.error(f"Failed to load KB Data: {e}")

def get_kb_matches(query: str):
    query_tokens = set(query.lower().split())
    if not query_tokens: return []
    
    scored_articles = []
    for cat in KB_DATA_CACHE:
        for article in cat.get("articles", []):
            q_text = article["q"].lower()
            # simple token overlap
            q_tokens = set(q_text.split())
            overlap = len(query_tokens.intersection(q_tokens))
            score = overlap / len(query_tokens) if len(query_tokens) > 0 else 0
            scored_articles.append({"score": score, "article": article})
            
    scored_articles.sort(key=lambda x: x["score"], reverse=True)
    return scored_articles[:2]
"""
content = content.replace(old_chat_req, new_chat_req)

# 2. Update process_chat
# Using regular expressions to find the entire process_chat function
start_def = "async def process_chat(req: ChatRequest, auth: str = Depends(verify_api_key)):"
end_def = '@app.post("/analyze-chunks")'

start_idx = content.find(start_def)
end_idx = content.find(end_def, start_idx)

if start_idx != -1 and end_idx != -1:
    new_process_chat = """async def process_chat(req: ChatRequest, auth: str = Depends(verify_api_key)):
    session_id = req.session_id or "default"
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        
    history = chat_sessions[session_id]
    
    # 1. Structured Case Context Truncated
    context = ""
    case_summary_fact = "No active case."
    if req.case_id:
        c = get_case(req.case_id)
        if c:
            # Parse Bio/Deep features just to grab a couple top keys if we want, but let's just stick to structured core
            try:
                bio = json.loads(c['bio_features'])
                top_bio = {k: round(v, 4) for k, v in list(bio.items())[:3]}
            except:
                top_bio = {}
                
            context = f"Active Case: {c['case_id']} | Verdict: {c['verdict']} | Score: {c['fraud_score']}% | Bio Flags: {top_bio}"
            case_summary_fact = f"Active Case {c['case_id']} is {c['verdict']} ({c['fraud_score']}%)."
            
    # 2. KB Retrieval & Direct Cache Check
    kb_context = ""
    matches = get_kb_matches(req.message)
    if matches and matches[0]["score"] > 0.85:
        # High confidence exact match -> Skip LLM entirely
        ans = matches[0]["article"]["a"]
        # Add tracking note to history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": ans})
        if len(history) > 6:
            history = history[-6:]
        chat_sessions[session_id] = history
        return {"status": "success", "reply": ans, "model": "Direct KB Cache"}
        
    if matches:
        kb_text = []
        for m in matches:
            if m["score"] > 0.1: # Only inject if slightly relevant
                a_trunc = m["article"]["a"][:300] + "..." if len(m["article"]["a"]) > 300 else m["article"]["a"]
                kb_text.append(f"Q: {m['article']['q']}\\nA: {a_trunc}")
        if kb_text:
            kb_context = "\\n\\nBANK KNOWLEDGE BASE REFERENCE:\\n" + "\\n---\\n".join(kb_text)

    # 3. Rolling History Summarization
    # Keep last 2-3 turns (4-6 messages), summarize older ones
    # To keep it completely deterministic, the summary is just a list of past user topics
    if len(history) > 6:
        # We need to drop the oldest ones and update a deterministic summary
        old_msgs = history[:-6]
        history = history[-6:]
        chat_sessions[session_id] = history
        
    # Extract topics from all past user messages in history for the summary line
    past_topics = [msg["content"] for msg in history if msg["role"] == "user"]
    topic_summary = ""
    if past_topics:
        # Just list them briefly
        topics_str = " | ".join([t[:30] + ("..." if len(t)>30 else "") for t in past_topics[:-1]]) if len(past_topics) > 1 else "None"
        topic_summary = f"\\n\\nPrior Topics Discussed: {topics_str}\\n{case_summary_fact}"

    sys_prompt = f\"\"\"You are the VoiceGuard Forensic AI Agent exclusively built for UCO Bank's Fraud Control Cell. You analyze highly sensitive voice biometric data.
    
    CRITICAL INSTRUCTIONS:
    1. Act strictly as the VoiceGuard AI Agent. Keep answers professional and highly analytical.
    2. If an Active Case is provided, directly reference its Verdict and Bio Flags.
    3. Use the BANK KNOWLEDGE BASE REFERENCE if provided to answer policy/glossary questions. Do NOT invent facts.
    
    {context}{kb_context}{topic_summary}\"\"\"
    
    import requests
    try:
        headers = {
            "Authorization": "Bearer YOUR_OPENROUTER_API_KEY",
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "VoiceGuard AI Agent",
            "Content-Type": "application/json"
        }
        
        # Build messages payload
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": req.message})
        
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": messages
        }
        
        used_model = "gpt-4o-mini"
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_err:
            print(f"GPT-4o-mini failed ({req_err}), falling back to Gemini...")
            used_model = "gemini-2.0-flash-exp:free"
            payload["model"] = "google/gemini-2.0-flash-exp:free"
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
        data = response.json()
        reply_text = data["choices"][0]["message"]["content"]
        
        # Append to history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": reply_text})
        chat_sessions[session_id] = history
        
        return {"status": "success", "reply": reply_text, "model": used_model}
    except Exception as e:
        print("LLM Error:", e)
        return {"status": "error", "reply": "OpenRouter LLM Error: " + str(e), "model": "error"}

"""
    content = content[:start_idx] + new_process_chat + "\n" + content[end_idx:]

with codecs.open(html_path, "w", "utf-8") as f:
    f.write(content)

print("SUCCESS")
