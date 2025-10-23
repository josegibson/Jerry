"""
Log Database Manager

Handles SQLite storage for agent logs, events, and token usage tracking.
Replaces file-based logging with structured database storage.
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager


class LogDatabase:
    """
    Manages SQLite database for agent logs and monitoring.
    
    Schema:
        events: id, session_id, event_type, level, message, data, timestamp
        token_usage: id, session_id, provider, input_tokens, output_tokens, total_tokens, timestamp
        errors: id, session_id, error_type, message, traceback, timestamp
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the log database.
        
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
        conn.row_factory = sqlite3.Row
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
            
            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    level TEXT DEFAULT 'INFO',
                    message TEXT,
                    data TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Token usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    provider TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Errors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    error_type TEXT NOT NULL,
                    message TEXT,
                    traceback TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON events(event_type, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_session 
                ON events(session_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_token_usage_session 
                ON token_usage(session_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_session 
                ON errors(session_id, timestamp)
            """)
    
    def log_event(self, event_type: str, data: Dict[str, Any], 
                  session_id: Optional[str] = None, level: str = "INFO"):
        """
        Log an event to the database.
        
        Args:
            event_type: Type of event (e.g., "agent_startup", "tool_call")
            data: Event data dictionary
            session_id: Optional session identifier
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        message = data.get('message', '')
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (session_id, event_type, level, message, data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                event_type,
                level,
                message,
                json.dumps(data),
                datetime.now().isoformat()
            ))
    
    def log_token_usage(self, provider: str, usage_data: Dict[str, int], 
                       session_id: Optional[str] = None):
        """
        Log token usage to the database.
        
        Args:
            provider: LLM provider name
            usage_data: Dictionary with input_tokens, output_tokens, total_tokens
            session_id: Optional session identifier
        """
        input_tokens = usage_data.get('input_tokens', 0)
        output_tokens = usage_data.get('output_tokens', 0)
        total_tokens = usage_data.get('total_tokens', input_tokens + output_tokens)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO token_usage 
                (session_id, provider, input_tokens, output_tokens, total_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                provider,
                input_tokens,
                output_tokens,
                total_tokens,
                datetime.now().isoformat()
            ))
    
    def log_error(self, error_type: str, message: str, traceback: Optional[str] = None,
                  session_id: Optional[str] = None):
        """
        Log an error to the database.
        
        Args:
            error_type: Type of error
            message: Error message
            traceback: Optional traceback string
            session_id: Optional session identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO errors (session_id, error_type, message, traceback, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                error_type,
                message,
                traceback,
                datetime.now().isoformat()
            ))
    
    def get_events(self, session_id: Optional[str] = None, 
                   event_type: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve events from the database.
        
        Args:
            session_id: Filter by session (optional)
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                event = dict(row)
                event['data'] = json.loads(event['data']) if event['data'] else {}
                events.append(event)
            
            return events
    
    def get_token_usage_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get token usage summary.
        
        Args:
            session_id: Filter by session (optional)
            
        Returns:
            Dictionary with aggregated token usage
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT 
                        provider,
                        SUM(input_tokens) as total_input,
                        SUM(output_tokens) as total_output,
                        SUM(total_tokens) as total,
                        COUNT(*) as call_count
                    FROM token_usage
                    WHERE session_id = ?
                    GROUP BY provider
                """, (session_id,))
            else:
                cursor.execute("""
                    SELECT 
                        provider,
                        SUM(input_tokens) as total_input,
                        SUM(output_tokens) as total_output,
                        SUM(total_tokens) as total,
                        COUNT(*) as call_count
                    FROM token_usage
                    GROUP BY provider
                """)
            
            by_provider = {}
            total_input = 0
            total_output = 0
            total_all = 0
            
            for row in cursor.fetchall():
                provider = row['provider']
                by_provider[provider] = {
                    'input': row['total_input'] or 0,
                    'output': row['total_output'] or 0,
                    'total': row['total'] or 0,
                    'calls': row['call_count'] or 0
                }
                total_input += by_provider[provider]['input']
                total_output += by_provider[provider]['output']
                total_all += by_provider[provider]['total']
            
            return {
                'session_total': {
                    'input': total_input,
                    'output': total_output,
                    'total': total_all
                },
                'by_provider': by_provider
            }
    
    def get_errors(self, session_id: Optional[str] = None, 
                   limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve errors from the database.
        
        Args:
            session_id: Filter by session (optional)
            limit: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT * FROM errors 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM errors 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_logs(self, days: int = 30):
        """
        Delete logs older than specified number of days.
        
        Args:
            days: Number of days to keep
        """
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        cutoff_str = cutoff_date.isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_str,))
            cursor.execute("DELETE FROM token_usage WHERE timestamp < ?", (cutoff_str,))
            cursor.execute("DELETE FROM errors WHERE timestamp < ?", (cutoff_str,))
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
