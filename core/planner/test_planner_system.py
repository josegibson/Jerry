
import pytest
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Literal

# Assuming PlannerSystem is in the same directory or accessible via sys.path
from core.planner.planner_system import PlannerSystem

@pytest.fixture
def temp_planner_data_path(tmp_path):
    """Provides a temporary path for planner_data.json for each test."""
    return tmp_path / "planner_data.json"

@pytest.fixture
def planner_system(temp_planner_data_path):
    """Provides a fresh PlannerSystem instance for each test."""
    return PlannerSystem(temp_planner_data_path)

class TestPlannerSystem:

    def test_initialization_empty_file(self, planner_system, temp_planner_data_path):
        """Test that PlannerSystem initializes with an empty schedule if file doesn't exist."""
        assert not temp_planner_data_path.exists()
        assert planner_system.schedule == []

    def test_initialization_existing_file(self, temp_planner_data_path):
        """Test that PlannerSystem loads an existing schedule correctly."""
        initial_schedule = [
            {"task_id": str(uuid.uuid4()), "description": "Task 1", "due_time": (datetime.now() + timedelta(days=1)).isoformat(), "priority": 1, "status": "scheduled", "agent_id": "test", "created_at": datetime.now().isoformat(), "recurrence_pattern": None, "recurrence_interval": 1, "metadata": {}},
            {"task_id": str(uuid.uuid4()), "description": "Task 2", "due_time": (datetime.now() + timedelta(hours=1)).isoformat(), "priority": 2, "status": "scheduled", "agent_id": "test", "created_at": datetime.now().isoformat(), "recurrence_pattern": None, "recurrence_interval": 1, "metadata": {}},
        ]
        with open(temp_planner_data_path, "w", encoding="utf-8") as f:
            json.dump(initial_schedule, f, indent=2)
        
        planner = PlannerSystem(temp_planner_data_path)
        assert len(planner.schedule) == 2
        assert planner.schedule[0]["description"] == "Task 2" # Should be sorted by due_time, then priority
        assert planner.schedule[1]["description"] == "Task 1"

    def test_initialization_corrupted_file(self, temp_planner_data_path, capsys):
        """Test that PlannerSystem handles corrupted JSON file gracefully."""
        temp_planner_data_path.write_text("{\"invalid json")
        planner = PlannerSystem(temp_planner_data_path)
        assert planner.schedule == []
        captured = capsys.readouterr()
        assert "Warning: Planner data file" in captured.out

    def test_schedule_non_recurring(self, planner_system):
        """Test adding a single non-recurring task."""
        due_time = datetime.now() + timedelta(days=1)
        task_details = {"agent_id":"agent1", "description":"Buy groceries", "due_time":due_time, "priority":5}
        task = planner_system.schedule(task_details)
        assert len(planner_system.schedule) == 1
        assert task["description"] == "Buy groceries"
        assert task["due_time"] == due_time.isoformat()
        assert task["priority"] == 5
        assert task["status"] == "scheduled"
        assert task["recurrence_pattern"] is None

    def test_schedule_sorting(self, planner_system):
        """Test that tasks are sorted by due_time then priority."""
        now = datetime.now()
        planner_system.schedule({"agent_id":"a1", "description":"Task C", "due_time":now + timedelta(hours=2), "priority":1})
        planner_system.schedule({"agent_id":"a1", "description":"Task A", "due_time":now + timedelta(hours=1), "priority":2})
        planner_system.schedule({"agent_id":"a1", "description":"Task B", "due_time":now + timedelta(hours=1), "priority":1})

        assert planner_system.schedule[0]["description"] == "Task A"
        assert planner_system.schedule[1]["description"] == "Task B"
        assert planner_system.schedule[2]["description"] == "Task C"

    def test_schedule_recurring_daily(self, planner_system):
        """Test adding a daily recurring task."""
        due_time = datetime.now() + timedelta(hours=1)
        task = planner_system.schedule({"agent_id":"a1", "description":"Daily check", "due_time":due_time, "recurrence_pattern":"daily"})
        assert task["recurrence_pattern"] == "daily"
        assert task["recurrence_interval"] == 1

    def test_schedule_recurring_weekly(self, planner_system):
        """Test adding a weekly recurring task."""
        due_time = datetime.now() + timedelta(hours=1)
        task = planner_system.schedule({"agent_id":"a1", "description":"Weekly report", "due_time":due_time, "recurrence_pattern":"weekly", "recurrence_interval":2})
        assert task["recurrence_pattern"] == "weekly"
        assert task["recurrence_interval"] == 2

    def test_schedule_validation(self, planner_system):
        """Test input validation for schedule."""
        with pytest.raises(ValueError, match="due_time must be a datetime object."):
            planner_system.schedule({"agent_id":"a1", "description":"Invalid time", "due_time":"not_a_datetime"})
        with pytest.raises(ValueError, match="Invalid recurrence_pattern."):
            planner_system.schedule({"agent_id":"a1", "description":"Invalid recurrence", "due_time":datetime.now(), "recurrence_pattern":"yearly"})
        with pytest.raises(ValueError, match="recurrence_interval must be at least 1."):
            planner_system.schedule({"agent_id":"a1", "description":"Invalid interval", "due_time":datetime.now(), "recurrence_interval":0})

    def test_get_upcoming_tasks(self, planner_system):
        """Test retrieving tasks that are due."""
        now = datetime.now()
        task1 = planner_system.schedule({"agent_id":"a1", "description":"Past task", "due_time":now - timedelta(hours=1), "priority":1})
        task2 = planner_system.schedule({"agent_id":"a1", "description":"Due now", "due_time":now, "priority":2})
        task3 = planner_system.schedule({"agent_id":"a1", "description":"Future task", "due_time":now + timedelta(days=1), "priority":3})
        
        due_tasks = planner_system.getUpcomingTasks()
        assert len(due_tasks) == 2
        assert due_tasks[0]["description"] == "Due now"
        assert due_tasks[1]["description"] == "Past task"

    def test_mark_task_completed_non_recurring(self, planner_system):
        """Test marking a non-recurring task as completed."""
        task = planner_system.schedule({"agent_id":"a1", "description":"One-time job", "due_time":datetime.now() + timedelta(hours=1)})
        assert planner_system.mark_task_completed(task["task_id"])
        updated_task = next(t for t in planner_system.schedule if t["task_id"] == task["task_id"])
        assert updated_task["status"] == "completed"
        assert "completed_at" in updated_task
        assert len(planner_system.schedule) == 1 # No new recurrence

    def test_mark_task_completed_recurring(self, planner_system):
        """Test marking a recurring task as completed and new recurrence generation."""
        due_time = datetime.now() + timedelta(minutes=1)
        task = planner_system.schedule({"agent_id":"a1", "description":"Daily routine", "due_time":due_time, "recurrence_pattern":"daily"})
        
        assert planner_system.mark_task_completed(task["task_id"])
        
        # Verify original task is completed
        original_task = next(t for t in planner_system.schedule if t["task_id"] == task["task_id"] and t["status"] == "completed")
        assert original_task["status"] == "completed"
        assert "completed_at" in original_task

        # Verify new recurring task is added
        assert len(planner_system.schedule) == 2
        new_task = next(t for t in planner_system.schedule if t["task_id"] != task["task_id"])
        assert new_task["description"] == "Daily routine"
        assert new_task["status"] == "scheduled"
        assert datetime.fromisoformat(new_task["due_time"]) == datetime.fromisoformat(task["due_time"]) + timedelta(days=1)

    def test_mark_task_failed(self, planner_system):
        """Test marking a task as failed."""
        task = planner_system.schedule({"agent_id":"a1", "description":"Critical task", "due_time":datetime.now() + timedelta(hours=1)})
        assert planner_system.mark_task_failed(task["task_id"])
        updated_task = next(t for t in planner_system.schedule if t["task_id"] == task["task_id"])
        assert updated_task["status"] == "failed"
        assert "failed_at" in updated_task

    def test_mark_task_cancelled(self, planner_system):
        """Test marking a task as cancelled."""
        task = planner_system.schedule({"agent_id":"a1", "description":"Optional task", "due_time":datetime.now() + timedelta(hours=1)})
        assert planner_system.mark_task_cancelled(task["task_id"])
        updated_task = next(t for t in planner_system.schedule if t["task_id"] == task["task_id"])
        assert updated_task["status"] == "cancelled"
        assert "cancelled_at" in updated_task

    def test_update_task(self, planner_system):
        """Test updating properties of an existing task."""
        task = planner_system.schedule({"agent_id":"a1", "description":"Old description", "due_time":datetime.now() + timedelta(days=1), "priority":1})
        updated_task = planner_system.update_task(task["task_id"], description="New description", priority=10)
        assert updated_task["description"] == "New description"
        assert updated_task["priority"] == 10
        
        # Verify sorting after update
        planner_system.schedule({"agent_id":"a1", "description":"Another task", "due_time":datetime.now() + timedelta(days=1), "priority":5})
        updated_task_again = planner_system.update_task(task["task_id"], priority=20)
        assert planner_system.schedule[0]["task_id"] == updated_task_again["task_id"]

    def test_update_task_non_existent(self, planner_system):
        """Test updating a non-existent task."""
        assert planner_system.update_task(str(uuid.uuid4()), description="Non-existent") is None

    def test_get_all_tasks(self, planner_system):
        """Test retrieving all tasks."""
        planner_system.schedule({"agent_id":"a1", "description":"Task 1", "due_time":datetime.now()})
        planner_system.schedule({"agent_id":"a1", "description":"Task 2", "due_time":datetime.now()})
        assert len(planner_system.get_all_tasks()) == 2

