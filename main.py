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
info_collection_sessions: Dict[int, Dict[str, Any]] = {}  # user_id -> {"step": int, "data": {}}

# Enhanced learning cache
name_patterns = {}  # user_id -> {"name": "value", "patterns": [list of questions]}
keyword_index = defaultdict(list)  # keyword -> list of (user_id, fact_index) tuples

# Personal info collection questions
PERSONAL_INFO_QUESTIONS = [
    {"key": "name", "question": "What's your full name?"},
    {"key": "age", "question": "How old are you?"},
    {"key": "location", "question": "Where are you from? (city/country)"},
    {"key": "occupation", "question": "What do you do for work or study?"},
    {"key": "interests", "question": "What are your interests or hobbies?"},
    {"key": "goals", "question": "What are you hoping to learn or achieve?"}
]

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
            "topics": {},  # topic -> [related facts] for better organization
            "info_collected": False,  # whether personal info has been collected
            "info_declined": False,  # whether user declined to share info
            "age": None,
            "location": None,
            "occupation": None,
            "interests": None,
            "goals": None
        }
        save_db(data)
    else:
        # keep username up to date
        data["users"][uid]["username"] = user.username or user.first_name or data["users"][uid].get("username", "")
        # Ensure new fields exist
        fields_to_ensure = ["name", "topics", "info_collected", "info_declined", "age", "location", "occupation", "interests", "goals"]
        changed = False
        for field in fields_to_ensure:
            if field not in data["users"][uid]:
                if field in ["info_collected", "info_declined"]:
                    data["users"][uid][field] = False
                elif field == "topics":
                    data["users"][uid][field] = {}
                else:
                    data["users"][uid][field] = None
                changed = True
        if changed:
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
    Enhanced: Parses structured notes and personal information with better organization
    """
    ensure_user_stats_obj(user)
    data = load_db()
    uid = str(user.id)
    
    # Check if this text contains a name declaration
    extracted_name = extract_name_from_text(text)
    if extracted_name:
        data["users"][uid]["name"] = extracted_name
        logger.info(f"Stored name '{extracted_name}' for user {uid}")
    
    # Extract personal information from new users
    personal_info = extract_personal_info(text)
    if personal_info:
        for key, value in personal_info.items():
            data["users"][uid][key] = value
        logger.info(f"Stored personal info for user {uid}: {list(personal_info.keys())}")
    
    # Check if this is a structured note (like "Genetics – Short Note")
    topic_match = re.match(r'^([A-Za-z\s]+)\s*[–-]\s*(.+?)$', text.split('\n')[0])
    if topic_match:
        topic = topic_match.group(1).strip()
        note_type = topic_match.group(2).strip()
        
        # Store the entire note as a single structured entry
        structured_note = f"{topic}: {text}"
        if structured_note not in data["users"][uid]["personal_facts"]:
            data["users"][uid]["personal_facts"].append(structured_note)
        
        # Also parse individual bullet points and definitions
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                # Handle bullet points like "- Gene: A unit of heredity..."
                bullet_content = line[2:].strip()
                if ':' in bullet_content:
                    term, definition = bullet_content.split(':', 1)
                    structured_fact = f"{topic} - {term.strip()}: {definition.strip()}"
                    if structured_fact not in data["users"][uid]["personal_facts"]:
                        data["users"][uid]["personal_facts"].append(structured_fact)
            elif re.match(r'^\d+\.', line):
                # Handle numbered lists like Mendel's Laws
                if ':' in line or '–' in line:
                    structured_fact = f"{topic} - {line}"
                    if structured_fact not in data["users"][uid]["personal_facts"]:
                        data["users"][uid]["personal_facts"].append(structured_fact)
    else:
        # Regular text processing
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r"[.?!]\s*", text.strip())
        
        for sentence in sentences:
            s = sentence.strip()
            if len(s) > 5:
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

def extract_personal_info(text: str) -> Dict[str, str]:
    """Extract personal information like age, interests, etc."""
    info = {}
    text_lower = text.lower()
    
    # Extract age
    age_patterns = [
        r'i am (\d+) years? old',
        r'my age is (\d+)',
        r'age[:\s]+(\d+)',
        r'(\d+) years? old'
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text_lower)
        if match:
            info['age'] = match.group(1)
            break
    
    # Extract interests/hobbies
    interest_patterns = [
        r'my interests? (?:are?|include)[:\s]*([^.!?]+)',
        r'i (?:like|love|enjoy)[:\s]*([^.!?]+)',
        r'hobbies?[:\s]*([^.!?]+)',
        r'i study[:\s]*([^.!?]+)',
        r'studying[:\s]*([^.!?]+)'
    ]
    for pattern in interest_patterns:
        match = re.search(pattern, text_lower)
        if match:
            interests = match.group(1).strip()
            if 'interests' not in info:
                info['interests'] = interests
            else:
                info['interests'] += f", {interests}"
    
    return info

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
    """Check if the query is asking about any user's name or bot's name, return appropriate response"""
    data = load_db()
    query_lower = query.lower().strip()
    
    # Check if asking about bot's name (you/your refers to bot)
    bot_name_questions = [
        "what is your name",
        "what's your name",
        "who are you",
        "what are you called",
        "your name",
        "tell me your name",
        "what do you call yourself",
        "what should i call you"
    ]
    
    for pattern in bot_name_questions:
        if query_lower == pattern or query_lower == pattern + "?":
            return "I'm Tejas AI, created by Fᴛᴍ Dᴇᴠᴇʟᴏᴘᴇʀᴢ. I'm here to help you learn and remember information! shared by you."
    
    # Check all users for name matches
    for uid, user_data in data["users"].items():
        stored_name = user_data.get("name")
        if stored_name:
            name_lower = stored_name.lower()
            
            # Check if asking about this specific name
            if f"who is {name_lower}" in query_lower or f"who is {name_lower}?" in query_lower:
                if uid == str(user.id):
                    return f"You are {stored_name}."
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

def convert_personal_fact_pronouns(fact: str) -> str:
    """
    Convert personal facts from first person (my/me/I) to second person (your/you)
    for when the bot responds to the user about their own information
    """
    converted = fact
    
    # Convert pronouns - be careful with word boundaries
    pronoun_conversions = {
        r'\bmy\b': 'your',
        r'\bMe\b': 'You',
        r'\bme\b': 'you',
        r'\bI am\b': 'You are',
        r'\bi am\b': 'you are',
        r'\bI\b': 'You',
        r'\bmyself\b': 'yourself',
        r'\bMy\b': 'Your',
    }
    
    for pattern, replacement in pronoun_conversions.items():
        converted = re.sub(pattern, replacement, converted)
    
    return converted

def process_pronouns_for_context(text: str, user) -> str:
    """
    Process pronouns to understand context better:
    - 'you/your/yours' refers to the bot (Tejas AI)
    - 'my/me/myself' refers to the user
    """
    processed_text = text
    
    # Get user's name if available
    data = load_db()
    uid = str(user.id)
    user_name = "the user"
    if uid in data["users"] and data["users"][uid].get("name"):
        user_name = data["users"][uid]["name"]
    
    # Replace pronouns for better understanding
    # Note: This is for internal processing, not changing the user's message display
    return processed_text

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

def extract_specific_content(full_content: str, query: str) -> Optional[str]:
    """Extract specific part of structured content based on query"""
    query_lower = query.lower().strip()
    lines = full_content.split('\n')
    
    # Check if asking about a specific term (like "law of segregation")
    query_keywords = [word.strip('?.,!') for word in query_lower.split() if len(word) > 2]
    
    # Look for specific lines that match the query
    matching_lines = []
    for line in lines:
        line_lower = line.lower()
        
        # Check for exact phrase matches first
        if query_lower.replace('what is ', '').replace('what are ', '').replace('define ', '') in line_lower:
            # If it's a bullet point or numbered item, return just that part
            if line.strip().startswith(('-', '•')) or re.match(r'^\s*\d+\.', line):
                return line.strip()
            elif ':' in line:
                return line.strip()
        
        # Check for keyword matches in specific structures
        if any(keyword in line_lower for keyword in query_keywords if keyword not in ['what', 'is', 'are', 'the', 'of']):
            if line.strip().startswith(('-', '•')) or re.match(r'^\s*\d+\.', line) or ':' in line:
                matching_lines.append(line.strip())
    
    # If we found specific matching lines, return the best one
    if matching_lines:
        # Prefer exact matches or longer explanations
        best_match = max(matching_lines, key=len)
        return best_match
    
    # Check if asking about the main topic (like "genetics")
    first_line = lines[0] if lines else ""
    if any(keyword in first_line.lower() for keyword in query_keywords):
        # For main topic questions, return the definition part
        for line in lines[1:]:  # Skip the title
            if line.strip() and not line.strip().startswith(('-', '•', '1.', '2.', '3.')):
                if '.' in line and len(line.strip()) > 30:  # Likely a definition
                    return line.strip()
        
        # If no good definition found, return the first few lines
        definition_lines = []
        for line in lines[:4]:  # First few lines
            if line.strip() and not line.strip().startswith(('-', '•')):
                definition_lines.append(line.strip())
        
        if definition_lines:
            return '\n'.join(definition_lines)
    
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
    """Enhanced fact matching with specific support for structured notes"""
    if not facts:
        return None
    
    query_lower = query.lower().strip()
    
    # First, try to find specific sub-topic matches (like "law of segregation")
    for fact in facts:
        fact_lower = fact.lower()
        
        # Check for specific term matches within structured notes
        if any(term in fact_lower for term in query_lower.split() if len(term) > 2):
            # Look for specific definitions within the fact
            lines = fact.split('\n') if '\n' in fact else [fact]
            for line in lines:
                line_lower = line.lower()
                
                # Check if this line contains the specific term being asked about
                if all(word in line_lower for word in query_lower.split() if len(word) > 2):
                    # If it's a bullet point or numbered item, return just that part
                    if (line.strip().startswith('-') or line.strip().startswith('•') or 
                        re.match(r'^\s*\d+\.', line)):
                        return line.strip()
                    
                    # If it's part of a structured note, try to extract the relevant section
                    if ':' in line and any(word in line_lower for word in query_lower.split()):
                        return line.strip()
    
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
    
    query_lower = query.lower().strip()
    
    # Check specific personal info fields first (direct mapping)
    specific_info_checks = [
        (["age", "old", "how old"], "age", "You are {} years old."),
        (["name", "called", "who am i", "who are you"], "name", "Your name is {}."),
        (["from", "where", "location", "live"], "location", "You are from {}."),
        (["work", "job", "study", "occupation", "do"], "occupation", "You work/study: {}."),
        (["interest", "hobby", "like", "enjoy"], "interests", "Your interests are: {}."),
        (["goal", "want", "achieve", "hope"], "goals", "Your goals are: {}."),
    ]
    
    for keywords, field, template in specific_info_checks:
        if any(keyword in query_lower for keyword in keywords):
            field_value = user_data.get(field)
            if field_value:
                return template.format(field_value)
    
    # Check if this is a name question (legacy support)
    stored_name = user_data.get("name")
    if stored_name and is_name_question(query, stored_name):
        return f"Your name is {stored_name}."
    
    # Enhanced fact matching for personal_facts array
    best_match = find_best_fact_match(facts, query)
    if best_match:
        # Convert personal facts to use proper pronouns (my -> your)
        converted_fact = convert_personal_fact_pronouns(best_match)
        return converted_fact
    
    # Fallback to original fuzzy matching
    matches = get_close_matches(query, facts, n=1, cutoff=0.4)
    if matches:
        converted_fact = convert_personal_fact_pronouns(matches[0])
        return converted_fact
    
    return None

# ---------------- HANDLERS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    ensure_user_stats_obj(user)
    
    # Check if this is a new user and collect personal info
    data = load_db()
    uid = str(user.id)
    is_new_user = uid not in data["users"] or not data["users"][uid].get("info_collected", False)
    
    if is_new_user:
        # Ask if they want to share personal information
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("Yes, I'd like to share", callback_data="share_info_yes")],
            [InlineKeyboardButton("No thanks, maybe later", callback_data="share_info_no")]
        ])
        msg = (
            f"👋 Hello! I'm <b>Tejas AI</b> (Created by Fᴛᴍ Dᴇᴠᴇʟᴏᴘᴇʀᴢ)\n\n"
            f"Nice to meet you, {user.first_name}! I'm here to help you learn and remember information.\n\n"
            "Would you like to share some information about yourself so I can provide more personalized assistance?"
        )
        await update.message.reply_text(msg, parse_mode="HTML", reply_markup=kb)
    else:
        stored_name = data["users"][uid].get("name", user.first_name)
        msg = (
            f"👋 Welcome back, {stored_name}! I'm <b>Tejas AI</b> (Created by Fᴛᴍ Dᴇᴠᴇʟᴏᴘᴇʀᴢ)\n\n"
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

        # Handle callback data that may or may not have a payload
        if "|" in data:
            action, payload = data.split("|", 1)
        else:
            action = data
            payload = None
        if action == "teach_yes":
            pending_questions[user.id] = payload
            await query.message.reply_text(f"📝 Okay! Please send the answer for:\n\n❓ {payload}")
            await send_log(context, f"{user_repr(user)} agreed to teach answer for: {payload}")

        elif action == "teach_no":
            await query.message.reply_text("👍 Okay, skipping this question.")
            await send_log(context, f"{user_repr(user)} skipped teaching.")

        elif action == "share_info_yes":
            # Start personal info collection
            info_collection_sessions[user.id] = {"step": 0, "data": {}}
            first_question = PERSONAL_INFO_QUESTIONS[0]["question"]
            await query.message.reply_text(f"Great! Let's start with some basic information.\n\n{first_question}")
            await send_log(context, f"{user_repr(user)} started personal info collection.")

        elif action == "share_info_no":
            # Mark as declined and proceed normally
            data = load_db()
            uid = str(user.id)
            if uid in data["users"]:
                data["users"][uid]["info_collected"] = True
                data["users"][uid]["info_declined"] = True
                save_db(data)
            
            msg = (
                "No problem! You can always share information later if you change your mind.\n\n"
                "<b>Commands</b>:\n"
                "/start - This message\n"
                "/learn - Start learning mode\n"
                "/cancel - Stop learning mode\n\n"
                "How I work:\n"
                "- Ask me a question. I try to find exact matches, then fuzzy matches.\n"
                "- If I know the answer: I send text + audio.\n"
                "- If I don't know: I ask if you want to teach me.\n"
            )
            await query.message.reply_text(msg, parse_mode="HTML")
            await send_log(context, f"{user_repr(user)} declined personal info collection.")

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

        # Enhanced pronoun processing for better understanding
        text = process_pronouns_for_context(text, user)

        # Handle personal info collection sessions
        if user.id in info_collection_sessions:
            session = info_collection_sessions[user.id]
            step = session["step"]
            
            if step < len(PERSONAL_INFO_QUESTIONS):
                question_key = PERSONAL_INFO_QUESTIONS[step]["key"]
                session["data"][question_key] = text
                
                # Move to next question
                step += 1
                session["step"] = step
                
                if step < len(PERSONAL_INFO_QUESTIONS):
                    next_question = PERSONAL_INFO_QUESTIONS[step]["question"]
                    await msg.reply_text(f"Thanks! Next question:\n\n{next_question}")
                else:
                    # Finished collecting info
                    info_collection_sessions.pop(user.id)
                    
                    # Save all collected info
                    data = load_db()
                    uid = str(user.id)
                    for key, value in session["data"].items():
                        data["users"][uid][key] = value
                    data["users"][uid]["info_collected"] = True
                    save_db(data)
                    
                    # Add to personal facts for searchability
                    info_text = f"My name is {session['data'].get('name', 'not provided')}. I am {session['data'].get('age', 'unknown')} years old. I am from {session['data'].get('location', 'unknown')}. I work/study: {session['data'].get('occupation', 'not specified')}. My interests are: {session['data'].get('interests', 'not specified')}. My goals are: {session['data'].get('goals', 'not specified')}."
                    add_personal_facts_from_text(user, info_text)
                    
                    msg_text = (
                        "Perfect! I've learned about you. Here's what I remember:\n\n"
                        f"• Name: {session['data'].get('name', 'Not provided')}\n"
                        f"• Age: {session['data'].get('age', 'Not provided')}\n"
                        f"• Location: {session['data'].get('location', 'Not provided')}\n"
                        f"• Work/Study: {session['data'].get('occupation', 'Not provided')}\n"
                        f"• Interests: {session['data'].get('interests', 'Not provided')}\n"
                        f"• Goals: {session['data'].get('goals', 'Not provided')}\n\n"
                        "Now you can ask me questions and I'll try my best to help! I understand that when you say 'you/your/yours' you're referring to me (Tejas AI), and when you say 'my/me/myself' you're referring to yourself."
                    )
                    await msg.reply_text(msg_text)
                    await send_log(context, f"Completed personal info collection for {user_repr(user)}")
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
            full_answer = entry.get("answer", "")
            taught_by = entry.get("taught_by", {})
            taught_by_str = f"{taught_by.get('username','unknown')} ({taught_by.get('id','?')})"
            
            # Extract specific part if possible
            specific_answer = extract_specific_content(full_answer, text)
            final_answer = specific_answer if specific_answer else full_answer
            
            await msg.reply_text(f"💡 {final_answer}")  # Line Y
            # TTS
            try:
                bio = tts_bytes(final_answer)
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
    
    # Import and start web server in a separate thread
    try:
        from bot import run_web_server
        import threading
        
        # Start web server in background thread
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()
        print("🌐 Web server started on port 5000")
        
    except Exception as e:
        print(f"⚠️ Failed to start web server: {e}")
    
    # Start Telegram bot
    app = build_app()
    app.run_polling()
