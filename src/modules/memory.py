import sqlite3
import re
from datetime import datetime
from typing import List, Dict, Set
from src.logger import logger

class MemoryModule:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY, username TEXT, display_name TEXT,
                first_seen TIMESTAMP, last_seen TIMESTAMP, interaction_count INTEGER DEFAULT 0
            )''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, message TEXT, response TEXT,
                timestamp TIMESTAMP, channel_id TEXT, FOREIGN KEY (user_id) REFERENCES users (user_id)
            )''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, fact_key TEXT, fact_value TEXT,
                confidence REAL DEFAULT 1.0, timestamp TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (user_id)
            )''')
        conn.commit()
        conn.close()

    def add_user(self, user_id: str, username: str, display_name: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users
            (user_id, username, display_name, first_seen, last_seen, interaction_count)
            VALUES (?, ?, ?,
                COALESCE((SELECT first_seen FROM users WHERE user_id = ?), ?),
                ?,
                COALESCE((SELECT interaction_count FROM users WHERE user_id = ?), 0) + 1)
        ''', (user_id, username, display_name or username, user_id,
              datetime.now(), datetime.now(), user_id))
        conn.commit()
        conn.close()

    def save_conversation(self, user_id: str, message: str, response: str, channel_id: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_id, message, response, timestamp, channel_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, message, response, datetime.now(), channel_id))
        self._extract_facts(user_id, message, cursor)
        conn.commit()
        conn.close()

    def _extract_facts(self, user_id: str, message: str, cursor):
        fact_patterns_ru = {
            'name': ['меня зовут', 'я ', 'моё имя', 'зови меня'],
            'age': ['мне ', 'лет', 'года', 'год', 'возраст'],
            'location': ['я живу в', 'я из', 'нахожусь в', 'мой город'],
            'hobby': ['я люблю', 'мне нравится', 'увлекаюсь', 'моё хобби', 'обожаю'],
            'job': ['я работаю', 'моя работа', 'моя профессия', 'специальность']
        }
        fact_patterns = fact_patterns_ru
        message_lower = message.lower()

        extracted_fact_types_this_message = set()

        for fact_type, patterns in fact_patterns.items():
            if fact_type in extracted_fact_types_this_message:
                continue

            for pattern in patterns:
                if pattern in message_lower:
                    try:
                        start_idx = message_lower.find(pattern)
                        potential_value_text = message[start_idx + len(pattern):].lstrip()
                        match = re.match(r"([^.,;!?]+)", potential_value_text)
                        fact_value = match.group(1).strip() if match else potential_value_text[:30].strip()

                        if fact_value:
                             cursor.execute('''
                                 INSERT OR REPLACE INTO user_facts
                                 (user_id, fact_key, fact_value, timestamp) VALUES (?, ?, ?, ?)
                             ''', (user_id, fact_type, fact_value, datetime.now()))
                             logger.info(f"Extracted/Updated fact for user {user_id}: {fact_type} = {fact_value}")
                             extracted_fact_types_this_message.add(fact_type)
                             break
                    except Exception as e:
                        logger.warning(f"Error extracting fact for pattern '{pattern}': {e}")

    def get_user_context_string(self, user_id: str, limit: int = 5) -> str:
        """Get recent conversation context for user as a string (legacy, might still be useful for quick display)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT message, response FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
        conversations = cursor.fetchall()
        cursor.execute("SELECT fact_key, fact_value FROM user_facts WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        facts = cursor.fetchall()
        conn.close()

        context_str = f"User facts: {'; '.join([f'{k}: {v}' for k, v in facts])}\n"
        context_str += "Recent conversations:\n"
        for msg, resp in reversed(conversations):
            context_str += f"User: {msg}\nAssistant: {resp}\n"
        return context_str

    def get_user_context_for_api(self, user_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation context for user, formatted for OpenAI-compatible API"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT message, response FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (user_id, limit))
        conversations = cursor.fetchall()

        cursor.execute('''
            SELECT fact_key, fact_value FROM user_facts
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        facts = cursor.fetchall()
        conn.close()

        history_messages: List[Dict[str, str]] = []

        if facts:
            # Not adding directly to history here as it's cleaner to handle it in system prompt or similar,
            # but following original logic:
            pass

        for msg, resp in reversed(conversations):
            if msg:
                 history_messages.append({"role": "user", "content": msg})
            if resp:
                 history_messages.append({"role": "assistant", "content": resp})

        return history_messages

    def get_user_facts(self, user_id: str) -> List[tuple]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT fact_key, fact_value FROM user_facts WHERE user_id = ?", (user_id,))
        facts = cursor.fetchall()
        conn.close()
        return facts
