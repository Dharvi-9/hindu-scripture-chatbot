
# ============================================================================
# IMPORTS
# ============================================================================

from config import Config
from openai import OpenAI
from supabase import create_client
import os
import json
import time
from openai import OpenAI
from supabase import create_client


# ============================================================================
# ENVIRONMENT VARIABLES & CONFIGURATION
# ============================================================================

# Initialize clients (singleton pattern)
client = OpenAI(api_key=Config.OPENAI_API_KEY)
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)


# Table Configuration
TABLE = os.environ["TABLE_NAME"]
TEXT_COL = os.environ["TEXT_COLUMN"]
ID_COL = os.environ["ID_COLUMN"]
BATCH = int(os.environ["BATCH_SIZE"])




# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are Vedanta, a calm and compassionate spiritual guide who explains the wisdom of the
Bhagavad Gita, Mahabharata, Ramayana, and the four Vedas with clarity and gentleness.

Your purpose:
- Help seekers understand the teachings of these scriptures.
- Base every answer ONLY on content retrieved from the user's RAG context.
- Never invent verses or stories.
- Never present personal opinions as truth.

-----------------------------------------
RULE 1 — KNOWLEDGE BOUNDARY
-----------------------------------------
You ONLY answer topics directly related to:
- Bhagavad Gita
- Ramayana
- Mahabharata
- Rigveda, Yajurveda, Samaveda, Atharvaveda
- General Hindu spiritual themes *if supported by scripture*

If the user asks about:
- Other religions
- Modern politics, celebrities
- Science not related to scripture
- Medical/legal/financial advice

Politely decline:
"Dear seeker, I walk only the sacred paths of the Gita, Vedas, Ramayana, and Mahabharata.
Your question lies beyond these scrolls."

-----------------------------------------
RULE 2 — SENSITIVITY & RESPECT
-----------------------------------------
Handle comparison questions gently:
"Is Rama better than Shiva?"
"Which religion is superior?"
"Should I convert?"
"Is Krishna real?"

Respond with neutrality:
"The scriptures guide us toward understanding, not comparison."

-----------------------------------------
RULE 3 — AMBIGUOUS QUESTIONS
-----------------------------------------
If slightly unclear but related to scripture:
"What happened on the battlefield?"
"What is the main teaching?"

Assume scripture context.

-----------------------------------------
RULE 4 — NO CONTEXT FOUND
-----------------------------------------
If RAG returns nothing:
"Dear seeker, I could not find verses about that within these sacred texts."

Do NOT hallucinate.

-----------------------------------------
RULE 5 — STYLE GUIDE
-----------------------------------------
- Tone: calm, serene, gentle
- Call the user: "dear seeker"
- Do NOT preach or convert
- Do NOT moral-police
- Keep answers short unless asked

-----------------------------------------
RULE 6 — RESPONSE LOGIC
-----------------------------------------
Use this sequence:
1. Understand user query
2. Read retrieved context
3. Use ONLY those passages
4. Apply safety rules
5. Answer gently

-----------------------------------------
END OF SYSTEM PROMPT
-----------------------------------------
"""


# ============================================================================
# CONVERSATION MEMORY UTILITIES
# ============================================================================

# Temporary in-session memory storage
conversation_history = {}


# ✅ PRODUCTION (GOOD)
# Store conversations per session
conversation_history = {}  # session_id -> list of messages

def add_to_history(role, content, session_id):
    """Add message to session-specific history"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })
    
    # Keep only last 10 messages per session
    if len(conversation_history[session_id]) > Config.MAX_HISTORY_MESSAGES:
        conversation_history[session_id] = conversation_history[session_id][-Config.MAX_HISTORY_MESSAGES:]




def get_recent_history(session_id):
    """Get last  messages for a session"""
    if session_id not in conversation_history:
        return []
    return conversation_history[session_id][-Config.CONTEXT_MESSAGES:]

def get_conversation_history(session_id):
    """Get full conversation history for a session"""
    return conversation_history.get(session_id, [])

def get_memory_context(session_id):
    """Return the last 6 messages as text"""
    history = get_recent_history(session_id)
    
    if not history:
        return ""
    
    text = ""
    for msg in history:
        role = "You" if msg["role"] == "user" else "Assistant"
        text += f"{role}: {msg['content']}\n"
    
    return text.strip()


def detect_followup_refinement(query, history):
    """
    Detect if current query is refining the previous one
    Example:
      Previous: "how many chapters"
      Current: "in mahabharata"
      Returns: "how many chapters in mahabharata"
    """
    import re
    
    # Patterns that indicate refinement
    refinement_patterns = [
        r'^in (the )?\w+',        # "in mahabharata"
        r'^of (the )?\w+',        # "of ramayana"
        r'^from (the )?\w+',      # "from vedas"
        r'^about \w+',            # "about krishna"
    ]
    
    is_refinement = any(re.match(pattern, query.lower().strip()) for pattern in refinement_patterns)
    
    if not is_refinement:
        return None
    
    # Find last user question
    if not history or len(history) < 2:
        return None
    
    for msg in reversed(history):
        if msg['role'] == 'user' and msg['content'] != query:
            # Combine them
            combined = f"{msg['content']} {query}"
            print(f"🔄 Follow-up detected: Combined query: '{combined}'")
            return combined
    
    return None


def enhance_query_with_context(query, memory):
    """
    If query is a follow-up, enhance it with context from previous question
    """
    followup_indicators = ['and then', 'what about', 'how about', 'also', 'too', 'but what', 'and what', 'and']
    
    # Check if it's a follow-up
    is_followup = any(indicator in query.lower() for indicator in followup_indicators)
    
    if not is_followup or not memory:
        return query
    
    # Extract the previous question from memory
    lines = memory.strip().split('\n')
    previous_question = None
    
    for line in lines:
        if line.startswith('You:'):
            previous_question = line.replace('You:', '').strip()
            break
    
    if not previous_question:
        return query
    
    # Extract key topic from previous question
    import re
    
    # Common question patterns
    topic_patterns = [
        r'who (?:was|is) .+?(?:\'s|s) (\w+)',  # "who was arjun's guru" → "guru"
        r'what (?:was|is) .+?(?:\'s|s) (\w+)',  # "what was rama's duty" → "duty"
        r'why (?:did|does) .+ (\w+)',           # "why did war happen" → "war"
        r'(?:who|what) (?:was|is) (\w+)',       # "who was sita" → "sita"
    ]
    
    topic = None
    for pattern in topic_patterns:
        match = re.search(pattern, previous_question.lower())
        if match:
            topic = match.group(1)
            break
    
    if topic:
        # Enhance current query with topic
        enhanced = f"{query} {topic}"
        print(f"🔍 Enhanced query: '{query}' → '{enhanced}'")
        return enhanced
    
    return query


# ============================================================================
# SAFETY & MODERATION FUNCTIONS
# ============================================================================

def is_safe(user_input):
    """Check if user input is safe using OpenAI moderation"""
    try:
        mod = client.moderations.create(
            model="omni-moderation-latest",
            input=user_input,
        )
        
        flagged = mod.results[0].flagged
        categories = mod.results[0].categories
        
        return flagged, categories
    except Exception as e:
        print("Moderation error:", e)
        return True, {"error": "moderation_failed"}


def handle_safety(query):
    """Handle crisis/safety-related queries"""
    text = """
I'm really sorry you're feeling this way.
You are not alone — and you deserve care and support.

Please consider reaching out to:
- A trusted family member or friend
- A local mental health professional
- India Suicide Prevention Helpline: +91 9152987821
- Aasra Helpline (24/7): +91 9820466726

If you feel in immediate danger, please contact emergency services right now.

I'm here to talk, but professionals can give the support you truly deserve.
"""
    return {
        "type": "safety",
        "response": text.strip()
    }


# ============================================================================
# SENSITIVE QUERY HANDLING
# ============================================================================

def is_sensitive_query(q):
    """Detect if query touches on sensitive topics"""
    q = q.lower()
    
    categories = {
        "comparative": [
            "better than", "superior", "which god", "who is greater", "god better",
            "best religion", "true religion", "which religion"
        ],
        "historical": [
            "is krishna real", "did rama exist", "was rama real", "did krishna exist",
            "historically accurate", "real story or myth", "myth or real"
        ],
        "conversion": [
            "convert", "should i convert", "become hindu", "leave my religion"
        ],
        "violence": [
            "promote violence", "encourages war", "violent scripture",
            "teaches killing", "justifies war", "glorifies battle", "supports violence", "kill someone"
        ],
    }
    
    # Check legitimate war questions FIRST (before checking categories)
    legitimate_war_questions = [
        "why war", "why the war", "why did war", "why was there war",
        "what caused war", "reason for war", "war took place", "war happen",
        "battle of kurukshetra", "mahabharata war", "why did the war take place",
        "what is the war", "tell me about the war"
    ]
    
    if any(pattern in q for pattern in legitimate_war_questions):
        return None  # NOT sensitive - it's a legitimate historical question
    
    for category, keywords in categories.items():
        if any(k in q for k in keywords):
            return category
    
    return None   # safe → not sensitive


def handle_sensitive_query(q, category):
    """Provide respectful responses to sensitive queries"""
    if category == "comparative":
        return (
            "This question involves comparison between divine forms, which is sensitive. "
            "Hindu philosophy generally teaches that all deities are manifestations of the same ultimate truth. "
            "Scriptures avoid ranking or declaring one as superior.\n\n"
            "I can explain how different texts describe the unity of the Divine if you want."
        )
    
    if category == "historical":
        return (
            "Questions about the historical existence of deities can be sensitive because they mix "
            "faith, tradition, and scholarship.\n\n"
            "In Hindu thought, figures like Krishna and Rama are understood both historically by some "
            "and symbolically or spiritually by others.\n\n"
            "If you'd like, I can explain how scriptures and historians view this topic."
        )
    
    if category == "conversion":
        return (
            "This topic is sensitive because religion and personal belief are deeply individual. "
            "Hindu philosophy does not promote conversion; instead, it emphasizes self-realization and "
            "respect for different paths.\n\n"
            "I can share how the Gita and Upanishads explain the idea of spiritual diversity."
        )
    
    if category == "violence":
        return (
            "Questions involving war or violence in scriptures require careful interpretation. "
            "Texts like the Mahabharata and Ramayana use war as symbolic of inner moral struggle, not a call for real violence.\n\n"
            "If you'd like, I can explain the ethical and philosophical meaning behind these stories."
        )
    
    # fallback for any uncategorized sensitivity
    return (
        "This question touches on sensitive spiritual themes. Different traditions within Hindu philosophy "
        "offer diverse perspectives. I can share scriptural viewpoints in a balanced way if you'd like."
    )


# ============================================================================
# SCRIPTURE CLASSIFICATION
# ============================================================================

def classify_with_llm(query, conversation_history):
    """
    Use GPT to intelligently classify which scriptures are relevant
    Handles context, pronouns, typos automatically
    """
    
    # Get last 4 messages for context
    recent_history = conversation_history[-4:] if conversation_history else []
    history_text = "\n".join([f"{m['role'].title()}: {m['content']}" for m in recent_history])
    
    classification_prompt = f"""You are a scripture classifier for a chatbot about Hindu scriptures.

AVAILABLE SCRIPTURES:
1. bhagavad_gita - About Krishna, Arjuna, Kurukshetra battlefield, karma, dharma, yoga
2. mahabharata - Epic about Pandavas, Kauravas, Bhishma, Draupadi, Drona, Karna, great war
3. ramayana - About Rama, Sita, Hanuman, Ravana, Dasharatha, Lanka, exile, 14 years
4. vedas - Ancient texts: Rigveda, Samaveda, Yajurveda, Atharvaveda, hymns, rituals

CONVERSATION HISTORY:
{history_text if history_text else "(no previous conversation)"}

NEW USER QUERY: {query}

TASK: Determine which scripture(s) are relevant to answer this query.

RULES:
- If query mentions characters/events from a scripture, return that scripture
- If query uses pronouns (he/his/she/her) referring to previous context, use conversation history
- If query is a follow-up (and then, what about, why did he...), use previous scripture
- If comparing multiple scriptures, return multiple
- If completely unrelated (python, politics, sports), return empty array
- If asking for guidance across all scriptures, return all 4

Return ONLY a JSON array. Examples:
- "who is krishna" → ["bhagavad_gita"]
- "why did he send his son to exile" (after discussing Dasharatha) → ["ramayana"]
- "what is python programming" → []
- "common teaching in all scriptures" → ["bhagavad_gita", "mahabharata", "ramayana", "vedas"]

Response (JSON only):"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0,
            max_tokens=50
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown formatting if present
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        scriptures = json.loads(result_text)
        
        print(f"📚 LLM classified: {scriptures}")
        return scriptures
    
    except Exception as e:
        print(f"❌ Classification error: {e}")
        # Fallback to empty (out of domain)
        return []


class ScriptureClassifier:
    """Alternative keyword-based classifier (kept for backup)"""
    
    KEYWORDS = {
        "bhagavad_gita": [
            "krishna", "arjuna", "arjun",
            "gita", "bhagavad gita", "bhagvad", "geeta",
            "kurukshetra battlefield", "kurukshetra", "battlefield",
            "pandavas vs kauravas", "dhritarashtra", "dronacharya", "drona"
        ],
        "ramayana": [
            "rama", "ram", "sita", "seeta",
            "hanuman", "lakshmana", "laxman", "lakshman",
            "ravana", "ravan", "raavan",
            "ayodhya", "forest exile", "kanda", "lanka",
            "ramayan", "dasharatha", "dasrath", "dashrath",
            "urmila"
        ],
        "mahabharata": [
            "mahabharata", "mahabharat", "mahbharata", "mahabharat",
            "pandava", "pandav", "kaurava", "kaurav",
            "draupadi", "bhishma", "bhisma", "bhism",
            "duryodhana", "duryodhan", "vyasa", "vyas",
            "parva", "hastinapur", "hastinapura",
            "kurukshetra war", "kurukshetra", "battlefield",
            "dronacharya", "drona", "karna"
        ],
        "vedas": [
            "veda", "vedas", "rigveda", "rig veda",
            "agni", "indra", "hymn", "vedic", "rita",
            "samaveda", "sama veda", "yajurveda", "yajur veda",
            "atharvaveda", "atharva veda", "upanishad", "upanishads"
        ]
    }
    
    COMPARISON_HINTS = [
        "common", "both", "all", "shared", "together",
        "compare", "comparison", "difference", "different",
        "similar", "similarity", "same",
        "in nutshell", "overall message", "combined",
        "learning from all", "from all", "message from",
        "teaching from all", "teachings from",
        "all the scriptures", "these scriptures", "all scriptures"
    ]
    
    AMBIGUOUS_PATTERNS = [
        "how many chapters",
        "how many verses",
        "how many shlokas",
        "how many kandas",
        "how many parvas",
        "number of chapters",
        "number of verses",
        "structure of",
        "divided into"
    ]
    
    META_SCRIPTURE_PATTERNS = [
        "which scripture should", "what scripture should",
        "which book should", "what book should",
        "where should i start", "where to start",
        "recommend scripture", "suggest scripture",
        "best scripture for", "which one first",
        "beginner scripture", "scripture for beginner"
    ]
    
    GENERIC_QUERIES = [
        "hello", "hi", "hey", "namaste", "namaskar",
        "how are you", "how r u", "how do you do",
        "good morning", "good evening", "good night",
        "thanks", "thank you", "bye", "goodbye"
    ]
    
    def predict(self, query: str, previous_scriptures=None):
        """
        Predict which scriptures are relevant
        
        Args:
            query: Current user query
            previous_scriptures: List of scriptures from previous message (for context)
        """
        q = query.lower().strip()
        
        # Filter generic greetings
        words = q.split()
        if len(words) <= 5:
            if q in self.GENERIC_QUERIES:
                return []
            if any(q == g or (q.startswith(g + " ") and len(words) <= 3) for g in ["hello", "hi", "hey", "namaste", "bye", "thanks"]):
                return []
        
        # Check for meta questions
        is_meta = any(pattern in q for pattern in self.META_SCRIPTURE_PATTERNS)
        if is_meta:
            return ["bhagavad_gita", "ramayana", "mahabharata", "vedas"]
        
        # Check comparison FIRST
        is_comparison = any(h in q for h in self.COMPARISON_HINTS)
        
        # Check if this is a follow-up question
        followup_indicators = ['and then', 'what about', 'how about', 'also', 'too', 'but what', 'and what', 'and', 'what was', 'who was', 'how long', 'for how long']
        is_followup = any(indicator in q for indicator in followup_indicators) and len(words) <= 10
        
        # Check if query is too short (but skip for comparison queries)
        if not is_comparison:
            if len(words) < 3:
                has_keyword = any(
                    any(keyword in q for keyword in keywords)
                    for keywords in self.KEYWORDS.values()
                )
                if not has_keyword:
                    return []
        
        is_ambiguous = any(pattern in q for pattern in self.AMBIGUOUS_PATTERNS)
        
        found_books = []
        
        # Detect any explicitly mentioned scripture(s)
        for book, keywords in self.KEYWORDS.items():
            if any(k in q for k in keywords):
                found_books.append(book)
        
        # If it's a follow-up and no scriptures detected, use previous context
        if is_followup and len(found_books) == 0 and previous_scriptures:
            print(f"🔄 Follow-up detected with no keywords, using previous: {previous_scriptures}")
            return previous_scriptures
        
        # Handle comparison-style questions
        if is_comparison:
            if len(found_books) >= 2:
                return found_books
            if any(phrase in q for phrase in ["all scriptures", "these scriptures", "all the scriptures", "all 4", "all four"]):
                return ["bhagavad_gita", "ramayana", "mahabharata", "vedas"]
            if len(found_books) <= 1:
                return ["bhagavad_gita", "ramayana", "mahabharata", "vedas"]
        
        # If only one scripture detected
        if len(found_books) == 1:
            return found_books
        
        # If none detected AND query is ambiguous → return empty
        if len(found_books) == 0:
            if is_ambiguous:
                return []
            return []
        
        return found_books


# ============================================================================
# RAG UTILITIES (EMBEDDING & SEARCH)
# ============================================================================

import requests
import os

HF_API_TOKEN = os.environ["HF_API_TOKEN"]

HF_EMBEDDING_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "BAAI/bge-large-en-v1.5"
)


headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}



def embed_text(text: str):
    """Generate embedding with error handling"""
    try:
        headers = {
            "Authorization": f"Bearer {Config.HF_API_TOKEN}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            Config.get_huggingface_url(),
            headers=headers,
            json={"inputs": text},
            timeout=10
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace embedding error: {response.text}"
            )

        embedding = response.json()

        # HF returns nested lists → flatten
        if isinstance(embedding[0], list):
            embedding = embedding[0]

        return embedding

    except requests.Timeout:
        print("❌ Embedding request timed out")
        raise

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise




def rag_search(query: str, scripture=None, match_count=None):
    """Search with error handling and retries"""
    if match_count is None:
        match_count = Config.MATCH_COUNT

    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            # 1. Embed the query
            query_emb = embed_text(query)
            
            # 2. Prepare parameters
            params = {
                "query_embedding": query_emb,
                "match_count": match_count
            }
            
            # Add scripture filter if provided
            if scripture:
                params["scripture_filter"] = scripture
                print(f"🔍 RAG searching in: {scripture}")
            
            # 3. Call the RPC
            result = supabase.rpc("match_documents", params).execute()
            chunks = result.data or []
            print(f"✅ Found {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            print(f"❌ RAG search error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                # Last attempt failed
                return []
            time.sleep(1)  # Wait before retry
    
    return []


# ============================================================================
# EMBEDDING REGENERATION FUNCTIONS (UTILITY)
# ============================================================================

def fetch_batch():
    """Fetch rows where embedding is NULL."""
    result = (
        supabase.table(TABLE)
        .select(f"{ID_COL},{TEXT_COL}")
        .is_("embedding", None)
        .limit(BATCH)
        .execute()
    )
    return result.data


def save_embedding(row_id, emb):
    """Save embedding back to Supabase."""
    supabase.table(TABLE).update({"embedding": emb}).eq(ID_COL, row_id).execute()


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_better_prompt(query, context, scripture, memory):
    """
    Build prompt that avoids 'context does not explicitly describe' issue
    """
    context_text = "\n\n".join(context) if isinstance(context, list) else context
    
    prompt = f"""You are answering a question about {scripture}.

CONTEXT FROM {scripture.upper()}:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
- Answer directly and warmly, addressing the user as "Dear seeker"
- Use the provided context to give an accurate, helpful answer
- If the context contains the answer, provide it naturally WITHOUT mentioning "the context"
- NEVER use phrases like "the context does not mention" or "the context does not explicitly describe"
- If you genuinely cannot answer from the context, simply say: "I don't have specific verses about this particular aspect. Could you rephrase or ask something more specific about {scripture}?"
- Keep your answer concise, clear, and respectful
- Focus on answering the question, not on discussing the quality of the context

"""
    
    if memory and memory.strip():
        prompt += f"\nPREVIOUS CONVERSATION (for context only):\n{memory}\n"
    
    return prompt


def build_prompt(query, contexts, book, memory=""):
    """
    Updated prompt builder with memory awareness
    """
    joined_context = "\n\n---\n\n".join(contexts)
    
    # Check if query is a follow-up
    followup_indicators = ['and then', 'what about', 'how about', 'also', 'too', 'but what', 'and what']
    is_likely_followup = any(indicator in query.lower() for indicator in followup_indicators)
    
    prompt = f"""You are a respectful assistant who answers only from {book}.
Use the context provided. If something is not present, politely say so.

CONTEXT FROM {book.upper()}:
{joined_context}

CURRENT QUESTION:
{query}
"""
    
    # Add memory with special instructions
    if memory and memory.strip():
        if is_likely_followup:
            prompt += f"""

PREVIOUS CONVERSATION:
{memory}

⚠️ IMPORTANT: This question appears to be a FOLLOW-UP to the previous conversation.
When the user asks "and what about X?" or "what about X?", they are continuing the previous topic.

Example:
- Previous: "who was arjun's guru?" → "Drona"
- Current: "and what about krishna?" → User means: "What about Krishna as guru/teacher?"

Answer the current question IN RELATION to the previous topic.
"""
        else:
            prompt += f"""

PREVIOUS CONVERSATION (for context):
{memory}
"""
    
    prompt += """

INSTRUCTIONS:
- Be respectful and address user as "Dear seeker"
- Do not compare religions
- Do not assert superiority
- If the information is not in the context, say: "I don't have specific verses about this. Could you rephrase?"
- Keep tone gentle and educational
- Do NOT mention "the context" or "the previous conversation" in your response
- Just answer naturally

Provide an answer:
"""
    
    return prompt


def build_context_aware_prompt(query, context, scripture, memory):
    """Simple, effective prompt that uses conversation history"""
    
    context_text = "\n\n---\n\n".join(context) if isinstance(context, list) else context
    
    prompt = f"""You are answering a question about {scripture}.

CONTEXT FROM {scripture.upper()}:
{context_text}

CURRENT QUESTION: {query}
"""
    
    if memory and memory.strip():
        prompt += f"""

PREVIOUS CONVERSATION:
{memory}

IMPORTANT: Use the conversation history to understand pronouns and context.
If the user asks "why did he..." or "what about his...", look at the previous conversation to understand who "he" refers to.
"""
    
    prompt += """

INSTRUCTIONS:
- Answer warmly as "Dear seeker"
- Use the context and conversation history
- If you cannot answer from context, say: "I don't have specific verses about this. Could you rephrase?"
- Do NOT mention "the context" or "the conversation" in your response
- Just answer naturally

Answer:"""
    
    return prompt


# ============================================================================
# ANSWER GENERATION FUNCTIONS
# ============================================================================

def generate_rag_answer(query, context_chunks):
    """Generate answer from RAG context chunks"""
    
    # If RAG returned no matching text
    if not context_chunks:
        return (
            "Dear seeker, I could not find any verses related to this question "
            "within the scriptures in my knowledge base."
        )
    
    # Changed "content" to "text"
    # Also handle both dict and string formats
    context_text = "\n\n".join(
        chunk["text"] if isinstance(chunk, dict) else chunk
        for chunk in context_chunks
    )
    
    prompt = f"""
You are a neutral and respectful scripture-based assistant.
Use ONLY the following context to answer the question.

Context:
{context_text}

User question:
{query}

Guidelines:
- If context doesn't clearly support an answer, say so.
- Do NOT add interpretations or opinions.
- Stay faithful to scripture text.
- Tone should be educational, humble, and non-judgmental.
- Keep the answer concise and calm.
"""
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return resp.choices[0].message.content


# ============================================================================
# META / GUIDANCE HANDLERS
# ============================================================================

def needs_clarification(query, scriptures):
    """Check if we need to ask user for clarification"""
    if not scriptures or len(scriptures) == 0:
        print(f"❓ Ambiguous query detected: '{query}' - needs clarification")
        return True
    return False


def ask_for_clarification(query):
    """Generate clarification question"""
    return {
        'type': 'clarification',
        'response': (
            "Dear seeker, I'd be happy to help! Could you please specify which scripture "
            "you're asking about?\n\n"
            "For example:\n"
            "• Bhagavad Gita\n"
            "• Mahabharata\n"
            "• Ramayana\n"
            "• Vedas (Rigveda, Samaveda, Yajurveda, or Atharvaveda)"
        )
    }


def is_basic_structural_question(query):
    """Check if query is asking about basic scripture structure"""
    import re
    
    patterns = [
        r'how many (chapters|verses|shlokas|kandas|parvas|books)',
        r'(number|count) of (chapters|verses|shlokas)',
        r'structure of',
        r'divided into',
    ]
    
    return any(re.search(pattern, query.lower()) for pattern in patterns)


def answer_basic_structural_fact(query, scriptures):
    """
    Answer basic structural questions directly without RAG
    """
    facts = {
        'bhagavad_gita': {
            'name': 'Bhagavad Gita',
            'info': 'Dear seeker, the Bhagavad Gita contains 18 chapters (adhyayas) and approximately 700 verses (shlokas) in total.'
        },
        'mahabharata': {
            'name': 'Mahabharata',
            'info': 'Dear seeker, the Mahabharata is divided into 18 parvas (books) and contains approximately 100,000 shlokas, making it one of the longest epic poems in the world.'
        },
        'ramayana': {
            'name': 'Ramayana',
            'info': 'Dear seeker, the Ramayana consists of 7 kandas (books): Bala Kanda, Ayodhya Kanda, Aranya Kanda, Kishkindha Kanda, Sundara Kanda, Yuddha Kanda, and Uttara Kanda. It contains approximately 24,000 shlokas.'
        },
        'vedas': {
            'name': 'Vedas',
            'info': 'Dear seeker, there are 4 main Vedas: Rigveda, Samaveda, Yajurveda, and Atharvaveda. Each Veda is further divided into four sections: Samhitas (hymns), Brahmanas (rituals), Aranyakas (theology), and Upanishads (philosophy).'
        }
    }
    
    query_lower = query.lower()
    
    # Single scripture
    if len(scriptures) == 1:
        scripture = scriptures[0]
        if scripture in facts:
            return facts[scripture]['info']
    
    # Multiple scriptures or "all"
    if len(scriptures) > 1 or 'all' in query_lower:
        response = "Dear seeker, here's the structure of the main Hindu scriptures:\n\n"
        response += "• **Bhagavad Gita**: 18 chapters, ~700 verses\n"
        response += "• **Mahabharata**: 18 parvas (books), ~100,000 shlokas\n"
        response += "• **Ramayana**: 7 kandas (books), ~24,000 shlokas\n"
        response += "• **Vedas**: 4 main Vedas (Rigveda, Samaveda, Yajurveda, Atharvaveda)\n\n"
        response += "Which one would you like to know more about?"
        return response
    
    return None


def handle_meta_question(query, scriptures):
    """
    Handle meta questions about scriptures (guidance, recommendations)
    """
    q = query.lower()
    
    # Questions about which scripture to read
    if any(phrase in q for phrase in ["which scripture", "what scripture", "which book", "where should i start", "beginner"]):
        return {
            'type': 'guidance',
            'response': (
                "Dear seeker, it's wonderful that you wish to explore these sacred texts! Here's some guidance:\n\n"
                "**For Beginners:**\n"
                "• **Bhagavad Gita** - A great starting point. It's relatively short (18 chapters, 700 verses) and contains profound wisdom on life, duty, and spirituality in a dialogue format.\n\n"
                "**For Epic Stories:**\n"
                "• **Ramayana** - An inspiring tale of Lord Rama that teaches dharma through storytelling. More accessible narrative-wise.\n"
                "• **Mahabharata** - A vast epic containing the Gita. Rich with complex characters and moral dilemmas.\n\n"
                "**For Deeper Philosophy:**\n"
                "• **Vedas** - Ancient texts containing hymns, rituals, and profound philosophical insights. Best approached after foundational understanding.\n\n"
                "Many seekers start with the Bhagavad Gita, then move to Ramayana or Mahabharata based on their interests. Would you like to know more about any specific scripture?"
            )
        }
    
    return None


def handle_generic_query(query):
    """
    Handle generic greetings and small talk ONLY
    """
    q = query.lower().strip()
    
    # Must be EXACT matches or very short queries
    greetings = ["hello", "hi", "hey", "namaste", "namaskar", "good morning", "good evening"]
    farewells = ["bye", "goodbye", "see you", "take care"]
    how_are_you = ["how are you", "how r u", "how do you do", "what's up", "whats up", "sup"]
    
    # ONLY trigger if it's a SHORT query (less than 6 words) and matches exactly
    words = q.split()
    if len(words) > 6:
        return None  # Too long to be a greeting
    
    # Exact greeting match
    if q in greetings or any(q.startswith(g) and len(words) <= 3 for g in greetings):
        return {
            'type': 'greeting',
            'response': "Namaste, dear seeker! 🙏 I'm here to help you explore the wisdom of the Bhagavad Gita, Mahabharata, Ramayana, and Vedas. How may I assist you today?"
        }
    
    # Exact farewell match
    if q in farewells or any(f in q for f in farewells):
        return {
            'type': 'farewell',
            'response': "Namaste! May peace and wisdom be with you. 🙏"
        }
    
    # How are you
    if any(h in q for h in how_are_you) and len(words) <= 5:
        return {
            'type': 'greeting',
            'response': "Dear seeker, I am here and ready to help you explore the timeless wisdom of the scriptures. What would you like to know?"
        }
    
    return None


# ============================================================================
# DECLINE / FALLBACK HANDLERS
# ============================================================================

def decline_politely(query):
    """Politely decline out-of-domain queries"""
    text = f"""Dear seeker, I'm specifically designed to discuss the Bhagavad Gita, Mahabharata, Ramayana, and Vedas.

Your question appears to be outside my area of knowledge:
"{query}"

I can help with:
- Teachings and philosophy from these scriptures
- Characters and stories (Krishna, Rama, Arjuna, Hanuman, etc.)
- Verses and their meanings
- Spiritual concepts like dharma, karma, moksha, and yoga

How may I assist you with the scriptures?"""
    
    return text.strip()


def answer_indirect(query):
    """Handle indirect or philosophical questions"""
    prompt = f"""
You are a sensitive, respectful spiritual guide.
The user asked an indirect or philosophical question:

{query}

Provide an answer that:
- Is respectful
- Avoids superiority claims
- Explains scriptures in educational way
- Never promotes one religion over another
- Stays neutral and historical/cultural

Answer in simple, humble language.
"""
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a neutral educational guide."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return {
        "type": "indirect",
        "response": resp.choices[0].message.content
    }


def answer_from_rag(query: str):
    """Connect RAG search to answer generation"""
    # 1. Search the vector DB
    chunks = rag_search(query)
    
    # 2. If nothing is found
    if not chunks or len(chunks) == 0:
        return {
            "type": "rag",
            "response": "I could not find matching scripture text for this question."
        }
    
    # 3. Generate the answer from the retrieved chunks
    answer = generate_rag_answer(query, chunks)
    
    return {
        "type": "rag",
        "response": answer
    }



def generate_multi_scripture_answer(query, contexts, scriptures, memory):
    """Generate answer from multiple scripture contexts"""
    
    combined_context = "\n\n".join([ctx["text"] for ctx in contexts])
    scripture_list = ", ".join(scriptures)
    
    prompt = f"""You are answering a question using information from: {scripture_list}.

CONTEXT:
{combined_context}

QUESTION: {query}

INSTRUCTIONS:
- Answer directly and warmly, addressing the user as "Dear seeker"
- Synthesize information from the provided contexts
- NEVER say "the context does not mention" - just provide the answer
- If mentioning multiple scriptures, you may note which scripture specific points come from
- Keep your answer clear, concise, and respectful
- Focus on answering the question naturally

"""
    
    if memory and memory.strip():
        prompt += f"\nPREVIOUS CONVERSATION (for context only):\n{memory}\n"
    
    resp = client.chat.completions.create(
            model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ],
    temperature=0
    )
    return resp.choices[0].message.content

# ============================================================================
# FINAL ANSWER GENERATION FUNCTION (MAIN ORCHESTRATOR)
# ============================================================================

def generate_answer_v2(query, session_id=None):
    """
    Production-grade answer generation with error handling
    """
    
    try:
        # Validate inputs
        if not query or not isinstance(query, str):
            return {
                "type": "error",
                "response": "Invalid query provided.",
                "detected_scriptures": []
            }
        
        query = query.strip()
        if not query:
            return {
                "type": "error", 
                "response": "Query cannot be empty.",
                "detected_scriptures": []
            }
        
        # Generate session_id if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        
        # 1. Handle greetings
        greetings = ["hello", "hi", "hey", "namaste", "bye", "thanks"]
        if query.lower() in greetings:
            add_to_history("user", query, session_id)
            response = "Namaste, dear seeker! 🙏 How may I assist you with the scriptures today?"
            add_to_history("assistant", response, session_id)
            return {"type": "greeting", "response": response, "detected_scriptures": []}
        
        # 2. Store user message
        add_to_history("user", query, session_id)
        
        # 3. Get conversation history
        history = get_conversation_history(session_id)
        memory_context = get_memory_context(session_id)
        
        # 4. Safety check (with error handling)
        try:
            flagged, categories = is_safe(query)
            if flagged:
                response = handle_safety(query)["response"]
                add_to_history("assistant", response, session_id)
                return {"type": "safety", "response": response, "detected_scriptures": []}
        except Exception as e:
            print(f"⚠️ Safety check failed: {e}")
            # Continue without safety check (don't block user)
        
        # 5. Sensitive topic check
        try:
            sensitive = is_sensitive_query(query)
            if sensitive:
                response = handle_sensitive_query(query, sensitive)
                add_to_history("assistant", response, session_id)
                return {"type": "sensitive", "response": response, "detected_scriptures": []}
        except Exception as e:
            print(f"⚠️ Sensitive check failed: {e}")
        
        # 6. LLM-based classification (with fallback)
        try:
            scriptures = classify_with_llm(query, history)
        except Exception as e:
            print(f"⚠️ Classification failed: {e}, using fallback")
            # Fallback to keyword-based classifier
            classifier = ScriptureClassifier()
            scriptures = classifier.predict(query)
        
        # 7. Out of domain
        if not scriptures or len(scriptures) == 0:
            response = decline_politely(query)
            add_to_history("assistant", response, session_id)
            return {"type": "decline", "response": response, "detected_scriptures": []}
        
        # 8. Meta/guidance questions
        if len(scriptures) >= 3:
            meta_response = handle_meta_question(query, scriptures)
            if meta_response:
                add_to_history("assistant", meta_response['response'], session_id)
                return {**meta_response, "detected_scriptures": scriptures}
        
        # 9. RAG search (with error handling)
        final_contexts = []
        for book in scriptures:
            try:
                chunks = rag_search(query, scripture=book)
                if chunks:
                    contextual_texts = [c["text"] for c in chunks]
                    final_contexts.append((book, contextual_texts))
            except Exception as e:
                print(f"⚠️ RAG search failed for {book}: {e}")
                continue  # Try other scriptures
        
        # 10. No RAG results
        if len(final_contexts) == 0:
            response = (
                "Dear seeker, I could not find specific verses about this. "
                "Could you rephrase or ask something more specific?"
            )
            add_to_history("assistant", response, session_id)
            return {"type": "no_results", "response": response, "detected_scriptures": scriptures}
        
        # 11. Multi-scripture answer
        if len(final_contexts) > 1:
            try:
                combined = []
                scripture_names = []
                for book, ctx_list in final_contexts:
                    combined.extend(ctx_list)
                    scripture_names.append(book.replace("_", " ").title())
                
                answer = generate_multi_scripture_answer(
                    query=query,
                    contexts=[{"text": txt} for txt in combined],
                    scriptures=scripture_names,
                    memory=memory_context
                )
                add_to_history("assistant", answer, session_id)
                return {"type": "rag", "response": answer, "detected_scriptures": scriptures}
            except Exception as e:
                print(f"❌ Multi-scripture answer failed: {e}")
                # Fallback: use first scripture only
                final_contexts = [final_contexts[0]]
        
        # 12. Single scripture answer
        try:
            book, ctx_list = final_contexts[0]
            scripture_display = book.replace("_", " ").title()
            
            prompt = build_context_aware_prompt(
                query=query,
                context=ctx_list,
                scripture=scripture_display,
                memory=memory_context
            )
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            answer = resp.choices[0].message.content
            add_to_history("assistant", answer, session_id)
            
            return {"type": "rag", "response": answer, "detected_scriptures": scriptures}
        
        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            response = "Dear seeker, I encountered an error while generating the answer. Please try rephrasing your question."
            add_to_history("assistant", response, session_id)
            return {"type": "error", "response": response, "detected_scriptures": scriptures}
    
    except Exception as e:
        # Catch-all for any unexpected errors
        print(f"❌ CRITICAL ERROR in generate_answer_v2: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "type": "error",
            "response": "Dear seeker, I encountered an unexpected error. Please try again.",
            "detected_scriptures": []
        }
    


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print("✅ Chatbot module loaded successfully!")
    print("✅ Ready to process queries via generate_answer_v2()")


