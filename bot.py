#!/usr/bin/env python3
"""
Merged: Advanced JSON-backed Knowledge Bot + Per-user fuzzy knowledge bot
Features:
- Persistent JSON DB (global knowledge with UUID qid)
- Per-user personal_facts list for quick fuzzy lookup
- Admin-managed global learning + personal learning sessions
- /start, /learn, /cancel, /edit, /get, /stats
- TTS replies and voice transcription
- Edit flow with inline callbacks
- Per-user stats tracked in DB
- Logging to console + Telegram log channel
"""

import os
import json
import uuid
import logging
import traceback
import re
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, Tuple, Optional, List
from difflib import get_close_matches
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# ---------------- CONFIG ----------------
BOT_TOKEN = "8052551435:AAF0zNhc-ViyRf8RPt_BHYAnq9MyaG2h0O4"
ADMINS = [7744665378, 6542804396]  # Add admin ids
LOG_CHANNEL = -1002807693664       # your log channel id
JSON_FILE = "ftm.json"

# ---------------- RUNTIME ----------------
pending_questions: Dict[int, str] = {}    # user_id -> question waiting for answer
personal_learn_sessions = set()           # user_ids in personal learn mode
edit_sessions: Dict[int, Dict[str, Any]] = {}  # user_id -> {"qid": qid, "mode": "q"/"a"}

# Enhanced learning cache
name_patterns = {}  # user_id -> {"name": "value", "patterns": [list of questions]}
keyword_index = defaultdict(list)  # keyword -> list of (user_id, fact_index) tuples

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FTM-KnowledgeBot")

async def send_log(context: ContextTypes.DEFAULT_TYPE, text: str):
    logger.info(text)
    try:
        await context.bot.send_message(LOG_CHANNEL, f"📜 <b>LOG</b>\n<pre>{text}</pre>", parse_mode="HTML")
    except Exception as e:
        logger.error(f"Failed send_log to LOG_CHANNEL: {e}")

# ---------------- DB ----------------
def init_db():
    if not os.path.exists(JSON_FILE):
        base = {
            "knowledge": {},  # qid -> {question, answer, taught_by, taught_at, edits:[]}
            "users": {},      # uid -> {username, stats:{messages,questions,answers,teaches,sessions}, personal_facts: []}
            "meta": {
                "global_learn": False,
                "global_learn_started_by": None
            }
        }
        with open(JSON_FILE, "w") as f:
            json.dump(base, f, indent=4)
    else:
        with open(JSON_FILE, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
        changed = False
        if "knowledge" not in data:
            data["knowledge"] = {}; changed = True
        if "users" not in data:
            data["users"] = {}; changed = True
        if "meta" not in data:
            data["meta"] = {"global_learn": False, "global_learn_started_by": None}; changed = True
        if changed:
            with open(JSON_FILE, "w") as f:
                json.dump(data, f, indent=4)

def load_db() -> Dict[str, Any]:
    with open(JSON_FILE, "r") as f:
        return json.load(f)

def save_db(data: Dict[str, Any]):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=4)

init_db()

# Initialize NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# ---------------- UTILITIES ----------------
def make_qid() -> str:
    return uuid.uuid4().hex

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def user_repr(user) -> str:
    uname = getattr(user, "username", None) or getattr(user, "first_name", None) or "unknown"
    return f"{uname} ({getattr(user, 'id', '??')})"

def safe_recognize_audio(src_path: str) -> Optional[str]:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(src_path) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception as e:
        logger.debug(f"Audio recognition failed: {e}")
        return None

def tts_bytes(text: str) -> BytesIO:
    tts = gTTS(text)
    bio = BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    return bio

# ---------------- DB OPERATIONS ----------------
def ensure_user_stats_obj(user) -> None:
    data = load_db()
    uid = str(user.id)
    if uid not in data["users"]:
        data["users"][uid] = {
            "username": user.username or user.first_name or "",
            "stats": {
                "messages": 0,
                "questions": 0,
                "answers": 0,
                "teaches": 0,
                "sessions": 0
            },
            "personal_facts": [],  # list of short strings (facts/sentences)
            "name": None,  # extracted name from conversations
            "topics": {}  # topic -> [related facts] for better organization
        }
        save_db(data)
    else:
        # keep username up to date
        data["users"][uid]["username"] = user.username or user.first_name or data["users"][uid].get("username", "")
        # Ensure new fields exist
        if "name" not in data["users"][uid]:
            data["users"][uid]["name"] = None
        if "topics" not in data["users"][uid]:
            data["users"][uid]["topics"] = {}
        save_db(data)

def increment_user_stat(user, stat: str, amount: int = 1):
    if stat not in ("messages", "questions", "answers", "teaches", "sessions"):
        return
    ensure_user_stats_obj(user)
    data = load_db()
    uid = str(user.id)
    data["users"][uid]["stats"][stat] = data["users"][uid]["stats"].get(stat, 0) + amount
    # ensure username saved
    data["users"][uid]["username"] = user.username or user.first_name or data["users"][uid].get("username", "")
    save_db(data)

def add_knowledge_entry(question: str, answer: str, user) -> str:
    """
    Add a new global knowledge entry; returns qid
    """
    data = load_db()
    qid = make_qid()
    data["knowledge"][qid] = {
        "question": question,
        "answer": answer,
        "taught_by": {
            "id": user.id,
            "username": user.username or user.first_name or ""
        },
        "taught_at": now_iso(),
        "edits": []
    }
    save_db(data)
    return qid

def update_knowledge_field(qid: str, field: str, new_value: str, editor):
    data = load_db()
    if qid not in data["knowledge"]:
        raise KeyError("qid-not-found")
    entry = data["knowledge"][qid]
    old = entry.get(field, "")
    entry[field] = new_value
    entry["edits"].append({
        "by": {"id": editor.id, "username": editor.username or editor.first_name or ""},
        "at": now_iso(),
        "field": field,
        "old": old
    })
    save_db(data)

def delete_knowledge(qid: str, deleter):
    data = load_db()
    if qid in data["knowledge"]:
        entry = data["knowledge"].pop(qid)
        save_db(data)
        return entry
    raise KeyError("qid-not-found")

# per-user personal_facts functions
def add_personal_facts_from_text(user, text: str):
    """
    Enhanced: Splits text into sentences, extracts names, and stores with better structure
    """
    ensure_user_stats_obj(user)
    data = load_db()
    uid = str(user.id)
    
    # Check if this text contains a name declaration
    extracted_name = extract_name_from_text(text)
    if extracted_name:
        data["users"][uid]["name"] = extracted_name
        logger.info(f"Stored name '{extracted_name}' for user {uid}")
    
    # Split into sentences for better organization  
    try:
        sentences = sent_tokenize(text)
    except:
        # Fallback if NLTK fails
        sentences = re.split(r"[.?!]\s*", text.strip())
    
    for sentence in sentences:
        s = sentence.strip()
        if len(s) > 5:  # Slightly longer minimum for quality
            # Avoid storing duplicate facts
            if s not in data["users"][uid]["personal_facts"]:
                data["users"][uid]["personal_facts"].append(s)
    
    save_db(data)

# ---------------- SEARCH / ANSWER LOGIC ----------------
def find_answer_by_question(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Enhanced global knowledge search with topic recognition
    1) exact case-insensitive match on global question
    2) exact match on global answer (in case they asked what's stored as a statement)
    3) topic-based matching (like "what is genetics" -> find genetics content)
    4) fuzzy match on global questions/answers
    5) return None otherwise (per-user fuzzy handled separately)
    """
    data = load_db()
    text_norm = text.strip().lower()
    
    # exact match on question
    for qid, entry in data["knowledge"].items():
        if entry.get("question", "").strip().lower() == text_norm:
            return qid, entry
    
    # exact match on answer text
    for qid, entry in data["knowledge"].items():
        if entry.get("answer", "").strip().lower() == text_norm:
            return qid, entry
    
    # Topic-based matching for global knowledge
    topic = is_topic_question(text)
    if topic:
        for qid, entry in data["knowledge"].items():
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            
            # Check if topic appears prominently in question or answer
            if (topic.lower() + ":") in question.lower() or (topic.lower() + ":") in answer.lower():
                return qid, entry
            
            # Check if question/answer starts with topic
            if question.lower().startswith(topic.lower()) or answer.lower().startswith(topic.lower()):
                return qid, entry
            
            # Check if topic appears as a significant keyword
            if topic.lower() in question.lower() or topic.lower() in answer.lower():
                return qid, entry
    
    # Enhanced fuzzy matching
    questions = [entry.get("question","") for entry in data["knowledge"].values()]
    answers = [entry.get("answer","") for entry in data["knowledge"].values()]
    
    # Check for keyword matches first (better for topic queries)
    query_words = set(text.lower().split())
    best_match = None
    best_score = 0
    
    for qid, entry in data["knowledge"].items():
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        
        # Score based on keyword overlap
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        q_overlap = len(query_words & q_words)
        a_overlap = len(query_words & a_words)
        total_overlap = q_overlap + a_overlap
        
        if total_overlap > best_score:
            best_score = total_overlap
            best_match = (qid, entry)
    
    if best_match and best_score > 0:
        return best_match
    
    # Fallback to original fuzzy matching
    q_matches = get_close_matches(text, questions, n=1, cutoff=0.4)
    if q_matches:
        for qid, entry in data["knowledge"].items():
            if entry.get("question","") == q_matches[0]:
                return qid, entry
    
    a_matches = get_close_matches(text, answers, n=1, cutoff=0.4)
    if a_matches:
        for qid, entry in data["knowledge"].items():
            if entry.get("answer","") == a_matches[0]:
                return qid, entry
    
    return None, None

def extract_name_from_text(text: str) -> Optional[str]:
    """Extract name from phrases like 'my name is John' or 'I am John'"""
    text_lower = text.lower().strip()
    
    # Common name patterns - improved to handle multi-word names
    patterns = [
        r'my name is ([a-zA-Z\s]+)',
        r'i am ([a-zA-Z\s]+)',
        r'call me ([a-zA-Z\s]+)',
        r'i\'?m ([a-zA-Z\s]+)',
        r'name is ([a-zA-Z\s]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).strip()
            # Handle multi-word names properly
            name_words = name.split()
            if len(name_words) <= 3:  # Reasonable limit for names
                return ' '.join(word.capitalize() for word in name_words)
    
    return None

def is_name_question(query: str, name: str) -> bool:
    """Check if query is asking about the stored name"""
    query_lower = query.lower().strip()
    name_lower = name.lower()
    
    # Direct name questions
    name_questions = [
        f"who is {name_lower}",
        f"what is my name",
        f"what's my name", 
        f"who am i",
        f"my name",
        f"tell me my name",
        f"do you know my name",
        f"what do you call me",
        f"who is {name_lower}?",
        f"what is {name_lower}"
    ]
    
    for pattern in name_questions:
        if query_lower == pattern or query_lower == pattern + "?":
            return True
    
    # Fuzzy matching for name questions
    if any(word in query_lower for word in ["name", "who", "call"]) and len(query_lower.split()) <= 5:
        return True
        
    return False

def check_name_question_for_user(query: str, user) -> Optional[str]:
    """Check if the query is asking about any user's name, return appropriate response"""
    data = load_db()
    query_lower = query.lower().strip()
    
    # Check all users for name matches
    for uid, user_data in data["users"].items():
        stored_name = user_data.get("name")
        if stored_name:
            name_lower = stored_name.lower()
            
            # Check if asking about this specific name
            if f"who is {name_lower}" in query_lower or f"who is {name_lower}?" in query_lower:
                if uid == str(user.id):
                    return f"{stored_name} is your name."
                else:
                    username = user_data.get("username", "someone")
                    return f"{stored_name} is the name of {username}."
    
    # Check if asking about own name
    uid = str(user.id)
    if uid in data["users"]:
        stored_name = data["users"][uid].get("name")
        if stored_name:
            # Questions about own name
            own_name_questions = [
                "what is my name",
                "what's my name",
                "who am i",
                "my name",
                "tell me my name",
                "do you know my name",
                "what do you call me"
            ]
            
            for pattern in own_name_questions:
                if query_lower == pattern or query_lower == pattern + "?":
                    return f"Your name is {stored_name}."
    
    return None

def extract_keywords_from_text(text: str) -> List[str]:
    """Extract meaningful keywords from text for better matching"""
    try:
        # Tokenize and remove stopwords
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
        return keywords
    except:
        # Fallback if NLTK fails
        words = text.lower().split()
        common_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        keywords = [word.strip('.,!?;:') for word in words if word not in common_words and len(word) > 2]
        return keywords

def is_topic_question(query: str) -> Optional[str]:
    """Check if query is asking 'what is [topic]' and extract the topic"""
    query_lower = query.lower().strip()
    
    # Common topic question patterns
    patterns = [
        r'what is ([a-zA-Z]+)\??',
        r'what are ([a-zA-Z]+)\??',
        r'define ([a-zA-Z]+)\??',
        r'explain ([a-zA-Z]+)\??',
        r'tell me about ([a-zA-Z]+)\??',
        r'([a-zA-Z]+) definition\??',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return match.group(1).capitalize()
    
    return None

def find_topic_definition(facts: List[str], topic: str) -> Optional[str]:
    """Find facts that define or describe a specific topic"""
    topic_lower = topic.lower()
    
    # Look for facts that start with the topic or have topic: format
    for fact in facts:
        fact_lower = fact.lower().strip()
        
        # Check if fact starts with "Topic:" format
        if fact_lower.startswith(f"{topic_lower}:"):
            return fact
            
        # Check if fact starts with the topic name
        if fact_lower.startswith(topic_lower):
            return fact
            
        # Check if fact contains topic definition patterns
        definition_patterns = [
            f"{topic_lower} is",
            f"{topic_lower} are", 
            f"{topic_lower}:",
            f"- {topic_lower}:"
        ]
        
        for pattern in definition_patterns:
            if pattern in fact_lower:
                return fact
    
    return None

def find_best_fact_match(facts: List[str], query: str) -> Optional[str]:
    """Enhanced fact matching using keywords and semantic similarity"""
    if not facts:
        return None
    
    # Check if this is a topic question first
    topic = is_topic_question(query)
    if topic:
        topic_fact = find_topic_definition(facts, topic)
        if topic_fact:
            return topic_fact
        
        # Also try with the original topic case
        original_topic = topic
        topic_fact = find_topic_definition(facts, original_topic)
        if topic_fact:
            return topic_fact
    
    query_keywords = set(extract_keywords_from_text(query))
    
    # Score facts based on keyword overlap
    scored_facts = []
    for fact in facts:
        fact_keywords = set(extract_keywords_from_text(fact))
        
        # Calculate various similarity scores
        keyword_overlap = len(query_keywords & fact_keywords)
        keyword_ratio = keyword_overlap / max(len(query_keywords), 1)
        
        # Substring match bonus
        substring_bonus = 2 if query.lower() in fact.lower() else 0
        
        # Topic match bonus - if any query word appears in fact
        topic_bonus = 0
        for word in query.lower().split():
            if len(word) > 2 and word in fact.lower():
                topic_bonus += 1
        
        # Length penalty for very long facts vs short queries
        length_penalty = max(0, 1 - abs(len(fact.split()) - len(query.split())) / 10)
        
        total_score = keyword_overlap + keyword_ratio + substring_bonus + topic_bonus + length_penalty
        
        if total_score > 0:
            scored_facts.append((fact, total_score))
    
    if scored_facts:
        # Sort by score and return best match
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        best_fact, best_score = scored_facts[0]
        
        # Lower threshold for better matching
        if best_score >= 0.5 or any(kw in best_fact.lower() for kw in query_keywords):
            return best_fact
    
    return None

def search_personal_facts(user, query: str) -> Optional[str]:
    data = load_db()
    uid = str(user.id)
    if uid not in data["users"]:
        return None
    
    user_data = data["users"][uid]
    facts = user_data.get("personal_facts", []) or []
    
    if not facts:
        return None
    
    # Check if this is a name question
    stored_name = user_data.get("name")
    if stored_name and is_name_question(query, stored_name):
        return f"Your name is {stored_name}."
    
    # Enhanced fact matching
    best_match = find_best_fact_match(facts, query)
    if best_match:
        return best_match
    
    # Fallback to original fuzzy matching
    matches = get_close_matches(query, facts, n=1, cutoff=0.4)
    if matches:
        return matches[0]
    
    return None

# ---------------- HANDLERS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    ensure_user_stats_obj(user)
    msg = (
        f"👋 Hello {user.first_name} (`{user.id}`)\n\n"
        "I am your unified FTM Knowledge Bot.\n\n"
        "<b>Commands</b>:\n"
        "/start - This message\n"
        "/learn - Start learning mode (admins -> global; users -> personal)\n"
        "/cancel - Stop learning mode\n"
        "/edit - Edit stored global knowledge\n"
        "/get - Download DB (admins only)\n"
        "/stats - Show leaderboard (admins only)\n\n"
        "How I work:\n"
        "- Ask a question. I try exact match, then fuzzy global, then fuzzy personal.\n"
        "- If I know: I send text + audio.\n"
        "- If unknown: I ask if you want to teach me.\n"
    )
    await update.message.reply_text(msg, parse_mode="HTML")
    await send_log(context, f"User started bot: {user_repr(user)}")

async def learn_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    If admin -> start global learn
    else -> start personal learn
    """
    user = update.effective_user
    if user.id in ADMINS:
        data = load_db()
        data["meta"]["global_learn"] = True
        data["meta"]["global_learn_started_by"] = {"id": user.id, "username": user.username or user.first_name}
        save_db(data)
        await update.message.reply_text("🟢 Global learning mode ENABLED. All messages will be stored as global knowledge until an admin sends /cancel.")
        await send_log(context, f"Global learning ENABLED by admin {user_repr(user)}")
    else:
        personal_learn_sessions.add(user.id)
        increment_user_stat(user, "sessions")
        await update.message.reply_text("🟢 Personal learning mode ENABLED for you. Everything you send will be stored in your personal facts until you send /cancel.")
        await send_log(context, f"Personal learning enabled for {user_repr(user)}")

async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    data = load_db()
    # admin cancels global learning if they choose
    if user.id in ADMINS:
        if data["meta"].get("global_learn"):
            data["meta"]["global_learn"] = False
            data["meta"]["global_learn_started_by"] = None
            save_db(data)
            await update.message.reply_text("🔴 Global learning mode DISABLED.")
            await send_log(context, f"Global learning DISABLED by admin {user_repr(user)}")
            return
    # personal cancel
    if user.id in personal_learn_sessions:
        personal_learn_sessions.discard(user.id)
        await update.message.reply_text("🔴 Personal learning mode DISABLED for you.")
        await send_log(context, f"Personal learning disabled for {user_repr(user)}")
    else:
        await update.message.reply_text("⚠️ You were not in any learning session. Admins can use /cancel to stop global learning.")

async def get_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id not in ADMINS:
        await update.message.reply_text("🚫 You are not authorized to use this command.")
        return
    await update.message.reply_document(JSON_FILE)
    await send_log(context, f"DB downloaded by admin {user_repr(user)}")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user.id not in ADMINS:
        await update.message.reply_text("🚫 You are not authorized to use this command.")
        return
    data = load_db()
    users = data.get("users", {})
    if not users:
        await update.message.reply_text("No user stats available yet.")
        return
    items = [(uid, info) for uid, info in users.items()]
    items.sort(key=lambda x: (x[1].get("stats",{}).get("teaches",0), x[1].get("stats",{}).get("questions",0)), reverse=True)
    text_lines = ["📊 <b>Leaderboard (top 10)</b>"]
    for i, (uid, info) in enumerate(items[:10], start=1):
        s = info.get("stats", {})
        text_lines.append(f"{i}. {info.get('username','unknown')} ({uid}) — teaches: {s.get('teaches',0)}, q: {s.get('questions',0)}, a: {s.get('answers',0)}")
    await update.message.reply_text("\n".join(text_lines), parse_mode="HTML")
    await send_log(context, f"Stats requested by admin {user_repr(user)}")

# ---------------- EDIT FLOW ----------------
def build_edit_list_markup():
    data = load_db()
    kb = []
    for idx, (qid, entry) in enumerate(data["knowledge"].items(), start=1):
        tb = entry.get("taught_by", {})
        tbname = tb.get("username", "") or ""
        qtext = entry.get("question", "")
        label = f"{idx}. {qtext} (by {tbname})"
        kb.append([InlineKeyboardButton(label[:70], callback_data=f"edit_select|{qid}")])
    if not kb:
        kb = [[InlineKeyboardButton("No entries", callback_data="noop")]]
    return InlineKeyboardMarkup(kb)

async def edit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = load_db()
    if not data["knowledge"]:
        await update.message.reply_text("📂 No global knowledge stored yet.")
        return
    await update.message.reply_text("📚 Select a global entry to edit:", reply_markup=build_edit_list_markup())

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = query.from_user
    data = query.data

    try:
        if data == "noop":
            await query.message.reply_text("Nothing to do.")
            return

        action, payload = data.split("|", 1)
        if action == "teach_yes":
            pending_questions[user.id] = payload
            await query.message.reply_text(f"📝 Okay! Please send the answer for:\n\n❓ {payload}")
            await send_log(context, f"{user_repr(user)} agreed to teach answer for: {payload}")

        elif action == "teach_no":
            await query.message.reply_text("👍 Okay, skipping this question.")
            await send_log(context, f"{user_repr(user)} skipped teaching.")

        elif action == "edit_select":
            qid = payload
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("Edit Question", callback_data=f"edit_q|{qid}")],
                [InlineKeyboardButton("Edit Answer", callback_data=f"edit_a|{qid}")],
                [InlineKeyboardButton("Delete Entry", callback_data=f"delete|{qid}")],
            ])
            entry = load_db()["knowledge"].get(qid)
            if entry:
                qtext = entry.get("question", "")
                atext = entry.get("answer", "")
                taught = entry.get("taught_by", {})
                taught_str = f"{taught.get('username','unknown')} ({taught.get('id','?')})"
                await query.message.reply_text(f"Selected entry:\n\nQ: {qtext}\nA: {atext}\nTaught by: {taught_str}", reply_markup=kb)
            else:
                await query.message.reply_text("Entry not found (maybe deleted).")

        elif action in ("edit_q", "edit_a"):
            qid = payload
            mode = "q" if action == "edit_q" else "a"
            edit_sessions[user.id] = {"qid": qid, "mode": mode}
            await query.message.reply_text("✏️ Send the new text now (it will overwrite).")
            await send_log(context, f"{user_repr(user)} started edit session for {qid} mode={mode}")

        elif action == "delete":
            qid = payload
            try:
                deleted = delete_knowledge(qid, user)
                await query.message.reply_text("🗑️ Entry deleted.")
                await send_log(context, f"{user_repr(user)} deleted entry: Q='{deleted.get('question','')}'")
            except KeyError:
                await query.message.reply_text("❌ Entry not found or already deleted.")
        else:
            await query.message.reply_text("Unknown action.")
    except Exception as e:
        err = f"Error in callback_handler: {e}\n{traceback.format_exc()}"
        logger.error(err)
        await send_log(context, err)

# ---------------- MESSAGE HANDLER ----------------
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = update.message
        user = update.effective_user
        text = None

        ensure_user_stats_obj(user)
        increment_user_stat(user, "messages")

        # voice support
        if msg.voice:
            vf = await msg.voice.get_file()
            ogg = f"tmp_{uuid.uuid4().hex}.ogg"
            wav = f"tmp_{uuid.uuid4().hex}.wav"
            await vf.download_to_drive(ogg)
            try:
                AudioSegment.from_file(ogg).export(wav, format="wav")
                recog = safe_recognize_audio(wav)
                if recog:
                    text = recog
                    # optional: show user's transcribed text
                    await msg.reply_text(f"🗣️ Transcribed: {recog}")
                else:
                    await msg.reply_text("❌ Sorry, could not transcribe the audio.")
            finally:
                if os.path.exists(ogg):
                    os.remove(ogg)
                if os.path.exists(wav):
                    os.remove(wav)

        elif msg.text:
            text = msg.text.strip()

        if not text:
            return

        # edit session handling (global knowledge editing)
        if user.id in edit_sessions:
            session = edit_sessions.pop(user.id)
            qid = session["qid"]
            mode = session["mode"]
            field = "question" if mode == "q" else "answer"
            try:
                update_knowledge_field(qid, field, text, user)
                await msg.reply_text(f"✅ Updated {field}.")
                await send_log(context, f"{user_repr(user)} edited {field} for qid={qid}")
            except KeyError:
                await msg.reply_text("❌ Entry not found. It may have been deleted.")
            return

        # If user had pending question (teach flow)
        if user.id in pending_questions:
            q = pending_questions.pop(user.id)
            qid = add_knowledge_entry(q, text, user)
            increment_user_stat(user, "teaches")
            await msg.reply_text(f"✅ Learned:\n\nQ: {q}\nA: {text}\nSaved as ID: {qid}")
            await send_log(context, f"Learned new QA by {user_repr(user)}: Q: {q} | A: {text} (qid={qid})")
            return

        # If global learn active -> store everything as global Q/A pair
        data = load_db()
        if data["meta"].get("global_learn"):
            # Extract name even in global learning mode
            extracted_name = extract_name_from_text(text)
            if extracted_name:
                ensure_user_stats_obj(user)
                data = load_db()
                uid = str(user.id)
                data["users"][uid]["name"] = extracted_name
                save_db(data)
                logger.info(f"Stored name '{extracted_name}' for user {uid} during global learning")
            
            qid = add_knowledge_entry(text, text, user)
            increment_user_stat(user, "teaches")
            await msg.reply_text("📌 Stored (global learning mode).")
            await send_log(context, f"GlobalLearn: Stored message from {user_repr(user)} as qid={qid}")
            return

        # Personal learn sessions: store text split into personal facts
        if user.id in personal_learn_sessions:
            add_personal_facts_from_text(user, text)
            increment_user_stat(user, "teaches")
            await msg.reply_text("📌 Stored to your personal facts (personal learning mode).")
            await send_log(context, f"PersonalLearn: Stored personal facts from {user_repr(user)}")
            return

        # Normal Q/A mode:
        increment_user_stat(user, "questions")
        
        # First check if this is a name question
        name_response = check_name_question_for_user(text, user)
        if name_response:
            await msg.reply_text(f"🧑‍💼 {name_response}")
            # TTS for name response
            try:
                bio = tts_bytes(name_response)
                await msg.reply_voice(bio)
            except Exception as e:
                logger.error(f"TTS failed on name response: {e}")
            increment_user_stat(user, "answers")
            await send_log(context, f"Answered name question for {user_repr(user)}: Q: {text}")
            return
        
        # Then check global knowledge
        qid, entry = find_answer_by_question(text)
        if entry:
            answer = entry.get("answer", "")
            taught_by = entry.get("taught_by", {})
            taught_by_str = f"{taught_by.get('username','unknown')} ({taught_by.get('id','?')})"
            await msg.reply_text(f"💡 {answer}\n\n👤 Taught by: {taught_by_str}")
            # TTS
            try:
                bio = tts_bytes(answer)
                await msg.reply_voice(bio)
            except Exception as e:
                logger.error(f"TTS failed: {e}")
            increment_user_stat(user, "answers")
            await send_log(context, f"Answered {user_repr(user)}: Q: {text} -> qid={qid}")
            return

        # fallback: try personal facts fuzzy search
        pf = search_personal_facts(user, text)
        if pf:
            await msg.reply_text(f"📖 (From your notes) {pf}")
            # TTS for personal fact
            try:
                bio = tts_bytes(pf)
                await msg.reply_voice(bio)
            except Exception as e:
                logger.error(f"TTS failed on personal fact: {e}")
            increment_user_stat(user, "answers")
            await send_log(context, f"Answered from personal facts for {user_repr(user)}: Q: {text}")
            return

        # Not known: offer to teach
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Yes, I'll teach", callback_data=f"teach_yes|{text}")],
            [InlineKeyboardButton("No thanks", callback_data="teach_no")]
        ])
        await msg.reply_text("🤔 I don't know this one. Can you teach me? (I can store globally)", reply_markup=kb)
        await send_log(context, f"Unknown question from {user_repr(user)}: {text}")
        return

    except Exception as e:
        err = f"Error in message_handler: {e}\n{traceback.format_exc()}"
        logger.error(err)
        try:
            await send_log(context, err)
            await update.message.reply_text("⚠️ An internal error occurred. Admins have been notified.")
        except Exception:
            pass

# ---------------- APP SETUP ----------------
def build_app() -> Application:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("learn", learn_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("get", get_cmd))
    app.add_handler(CommandHandler("edit", edit_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler((filters.TEXT | filters.VOICE) & ~filters.COMMAND, message_handler))
    return app

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🤖 FTM Knowledge Bot (merged) starting...")
    app = build_app()
    app.run_polling()
