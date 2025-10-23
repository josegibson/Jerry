"""
Session Database Manager

Handles SQLite storage for conversation sessions, messages, and metadata.
Replaces JSON-based state persistence with structured database storage.
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.load import dumpd, load


class SessionDatabase:
    """
    Manages SQLite database for agent sessions and conversation history.
    
    Schema:
        sessions: session_id, agent_name, started_at, ended_at, status
        messages: id, session_id, role, content, tool_calls, timestamp, message_data
        session_metadata: session_id, key, value
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the session database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT DEFAULT 'active',
                    message_count INTEGER DEFAULT 0
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    timestamp TEXT NOT NULL,
                    message_data TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Session metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    PRIMARY KEY (session_id, key),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_agent 
                ON sessions(agent_name, started_at)
            """)
    
    def create_session(self, session_id: str, agent_name: str) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Unique identifier for the session
            agent_name: Name of the agent
            
        Returns:
            The session_id
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, agent_name, started_at, status)
                VALUES (?, ?, ?, 'active')
            """, (session_id, agent_name, datetime.now().isoformat()))
        
        return session_id
    
    def add_message(self, session_id: str, message: BaseMessage):
        """
        Add a message to the session.
        
        Args:
            session_id: Session identifier
            message: LangChain message object
        """
        # Serialize the full message object
        message_data = json.dumps(dumpd(message))
        
        # Extract key fields for easy querying
        role = message.__class__.__name__
        content = getattr(message, 'content', '')
        tool_calls = None
        tool_call_id = None
        
        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls'):
            if message.tool_calls:
                tool_calls = json.dumps(message.tool_calls)
        
        if isinstance(message, ToolMessage):
            tool_call_id = getattr(message, 'tool_call_id', None)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages 
                (session_id, role, content, tool_calls, tool_call_id, timestamp, message_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                role,
                content,
                tool_calls,
                tool_call_id,
                datetime.now().isoformat(),
                message_data
            ))
            
            # Update message count
            cursor.execute("""
                UPDATE sessions 
                SET message_count = message_count + 1 
                WHERE session_id = ?
            """, (session_id,))
    
    def get_session_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Retrieve all messages for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of LangChain message objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_data 
                FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (session_id,))
            
            messages = []
            for row in cursor.fetchall():
                message_dict = json.loads(row['message_data'])
                messages.append(load(message_dict))
            
            return messages
    
    def close_session(self, session_id: str):
        """
        Mark a session as closed.
        
        Args:
            session_id: Session identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions 
                SET ended_at = ?, status = 'closed'
                WHERE session_id = ?
            """, (datetime.now().isoformat(), session_id))
    
    def clear_session(self, session_id: str):
        """
        Clear all messages from a session (for reset).
        
        Args:
            session_id: Session identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("""
                UPDATE sessions 
                SET message_count = 0 
                WHERE session_id = ?
            """, (session_id,))
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session info or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def list_sessions(self, agent_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent sessions.
        
        Args:
            agent_name: Filter by agent name (optional)
            limit: Maximum number of sessions to return
            
        Returns:
            List of session info dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if agent_name:
                cursor.execute("""
                    SELECT * FROM sessions 
                    WHERE agent_name = ?
                    ORDER BY started_at DESC 
                    LIMIT ?
                """, (agent_name, limit))
            else:
                cursor.execute("""
                    SELECT * FROM sessions 
                    ORDER BY started_at DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def set_metadata(self, session_id: str, key: str, value: Any):
        """
        Set a metadata value for a session.
        
        Args:
            session_id: Session identifier
            key: Metadata key
            value: Metadata value (will be JSON-serialized)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO session_metadata (session_id, key, value)
                VALUES (?, ?, ?)
            """, (session_id, key, json.dumps(value)))
    
    def get_metadata(self, session_id: str, key: str) -> Optional[Any]:
        """
        Get a metadata value for a session.
        
        Args:
            session_id: Session identifier
            key: Metadata key
            
        Returns:
            The metadata value or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM session_metadata 
                WHERE session_id = ? AND key = ?
            """, (session_id, key))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row['value'])
            return None
    
    def get_all_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Get all metadata for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of metadata key-value pairs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT key, value FROM session_metadata 
                WHERE session_id = ?
            """, (session_id,))
            
            return {row['key']: json.loads(row['value']) for row in cursor.fetchall()}
