import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal

# Assuming PlannerModule is accessible, e.g., passed during ToolProvider initialization
# For now, we'll define a placeholder for PlannerModule to allow tool definition.
# In a real scenario, the ToolProvider would instantiate and manage the PlannerModule.

class PlannerModulePlaceholder:
    def add_task(self, agent_id: str, description: str, due_time: datetime, priority: int = 0, recurrence_pattern: Optional[Literal["daily", "weekly", "monthly"]] = None, recurrence_interval: int = 1, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
    def get_due_tasks(self, current_time: datetime) -> List[Dict[str, Any]]:
        raise NotImplementedError
    def mark_task_completed(self, task_id: str) -> bool:
        raise NotImplementedError
    def mark_task_failed(self, task_id: str) -> bool:
        raise NotImplementedError
    def mark_task_cancelled(self, task_id: str) -> bool:
        raise NotImplementedError
    def update_task(self, task_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

# This will be replaced by an actual PlannerModule instance during runtime
_planner_module_instance: Optional[PlannerModulePlaceholder] = None

def _get_planner_module() -> PlannerModulePlaceholder:
    if _planner_module_instance is None:
        # In a real setup, this would be properly initialized by the ToolProvider
        # For now, raise an error if not set, to indicate improper usage.
        raise RuntimeError("PlannerModule not initialized for planner_tools.")
    return _planner_module_instance

def initialize_planner_tools(planner_module_instance: Any):
    global _planner_module_instance
    _planner_module_instance = planner_module_instance


def add_planner_task(agent_id: str, description: str, due_time_iso: str, priority: int = 0, recurrence_pattern: Optional[Literal["daily", "weekly", "monthly"]] = None, recurrence_interval: int = 1, metadata_json: Optional[str] = None) -> Dict[str, Any]:
    """
    Adds a new task to the planner.

    Args:
        agent_id: The ID of the agent requesting the task.
        description: A description of the task.
        due_time_iso: The initial due time of the task in ISO 8601 format (e.g., '2025-09-23T10:30:00').
        priority: The priority of the task (higher number = higher importance, default 0).
        recurrence_pattern: Optional pattern for recurring tasks ("daily", "weekly", "monthly").
        recurrence_interval: Interval for recurrence (e.g., 1 for every day/week/month, default 1).
        metadata_json: Optional JSON string for additional task-specific data (e.g., '{"project": "Jerry"}').

    Returns:
        A dictionary representing the added task.
    """
    due_time = datetime.fromisoformat(due_time_iso)
    metadata = json.loads(metadata_json) if metadata_json else None
    return _get_planner_module().add_task(agent_id, description, due_time, priority, recurrence_pattern, recurrence_interval, metadata)

def get_planner_due_tasks(current_time_iso: str) -> List[Dict[str, Any]]:
    """
    Retrieves tasks that are due at or before the specified current_time.

    Args:
        current_time_iso: The current time in ISO 8601 format (e.g., '2025-09-23T10:30:00').

    Returns:
        A list of dictionaries, each representing a due task.
    """
    current_time = datetime.fromisoformat(current_time_iso)
    return _get_planner_module().get_due_tasks(current_time)

def mark_planner_task_completed(task_id: str) -> bool:
    """
    Marks a specific task as completed.

    Args:
        task_id: The unique ID of the task to mark.

    Returns:
        True if the task was found and marked, False otherwise.
    """
    return _get_planner_module().mark_task_completed(task_id)

def mark_planner_task_failed(task_id: str) -> bool:
    """
    Marks a specific task as failed.

    Args:
        task_id: The unique ID of the task to mark.

    Returns:
        True if the task was found and marked, False otherwise.
    """
    return _get_planner_module().mark_task_failed(task_id)

def mark_planner_task_cancelled(task_id: str) -> bool:
    """
    Marks a specific task as cancelled.

    Args:
        task_id: The unique ID of the task to mark.

    Returns:
        True if the task was found and marked, False otherwise.
    """
    return _get_planner_module().mark_task_cancelled(task_id)

def update_planner_task(task_id: str, description: Optional[str] = None, due_time_iso: Optional[str] = None, priority: Optional[int] = None, recurrence_pattern: Optional[Literal["daily", "weekly", "monthly"]] = None, recurrence_interval: Optional[int] = None, metadata_json: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Updates an existing task's properties.

    Args:
        task_id: The unique ID of the task to update.
        description: Optional new description for the task.
        due_time_iso: Optional new due time in ISO 8601 format.
        priority: Optional new priority for the task.
        recurrence_pattern: Optional new recurrence pattern.
        recurrence_interval: Optional new recurrence interval.
        metadata_json: Optional new metadata as a JSON string.

    Returns:
        A dictionary representing the updated task, or None if the task was not found.
    """
    kwargs = {}
    if description is not None: kwargs["description"] = description
    if due_time_iso is not None: kwargs["due_time"] = datetime.fromisoformat(due_time_iso)
    if priority is not None: kwargs["priority"] = priority
    if recurrence_pattern is not None: kwargs["recurrence_pattern"] = recurrence_pattern
    if recurrence_interval is not None: kwargs["recurrence_interval"] = recurrence_interval
    if metadata_json is not None: kwargs["metadata"] = json.loads(metadata_json)

    return _get_planner_module().update_task(task_id, **kwargs)
