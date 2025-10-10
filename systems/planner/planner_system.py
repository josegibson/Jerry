import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Literal

class PlannerSystem:
    """
    A central, deterministic scheduling engine for agents to request and manage future tasks.
    This module is designed to be robust and self-contained, residing in its own directory.
    """

    def __init__(self, planner_data_path: Path):
        self.planner_data_path = planner_data_path
        self.tasks: List[Dict[str, Any]] = self._load_schedule()

    def _load_schedule(self) -> List[Dict[str, Any]]:
        """Loads the schedule from a JSON file."""
        if self.planner_data_path.exists():
            try:
                with open(self.planner_data_path, "r", encoding="utf-8") as f:
                    schedule = json.load(f)
                self.tasks = schedule
                self._sort_schedule() # Sort after loading
                return self.tasks
            except json.JSONDecodeError:
                print(f"Warning: Planner data file {self.planner_data_path} is corrupted. Starting with empty schedule.")
                return []
        return []

    def _save_schedule(self):
        """Saves the current schedule to a JSON file."""
        try:
            with open(self.planner_data_path, "w", encoding="utf-8") as f:
                json.dump(self.tasks, f, indent=2)
        except IOError as e:
            print(f"Error saving planner data to {self.planner_data_path}: {e}")

    def _sort_schedule(self):
        """Sorts the schedule by due_time, then by priority (descending)."""
        self.tasks.sort(key=lambda x: (datetime.fromisoformat(x['due_time']), -x['priority']))

    def schedule(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds a new task to the planner from a task details dictionary.

        Args:
            task_details: A dictionary containing the details of the task. 
                          Expected keys: 'agent_id', 'description', 'due_time', 
                          'priority', 'recurrence_pattern', 'recurrence_interval', 'metadata'.

        Returns:
            The added task dictionary.
        """
        agent_id = task_details.get("agent_id")
        description = task_details.get("description")
        due_time = task_details.get("due_time")
        priority = task_details.get("priority", 0)
        recurrence_pattern = task_details.get("recurrence_pattern")
        recurrence_interval = task_details.get("recurrence_interval", 1)
        metadata = task_details.get("metadata")

        if not all([agent_id, description, due_time]):
            raise ValueError("'agent_id', 'description', and 'due_time' are required in task_details.")

        if not isinstance(due_time, datetime):
            raise ValueError("due_time must be a datetime object.")
        if recurrence_pattern and recurrence_pattern not in ["daily", "weekly", "monthly"]:
            raise ValueError("Invalid recurrence_pattern. Must be 'daily', 'weekly', or 'monthly'.")
        if recurrence_interval < 1:
            raise ValueError("recurrence_interval must be at least 1.")

        task = {
            "task_id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "description": description,
            "due_time": due_time.isoformat(),
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "scheduled",
            "recurrence_pattern": recurrence_pattern,
            "recurrence_interval": recurrence_interval,
            "metadata": metadata if metadata is not None else {}
        }
        self.tasks.append(task)
        self._sort_schedule()
        self._save_schedule()
        return task

    def getUpcomingTasks(self) -> List[Dict[str, Any]]:
        """
        Retrieves tasks that are due at or before the current_time, sorted by due_time then priority.
        """
        current_time = datetime.now()
        due_tasks = []
        for task in self.tasks:
            if task["status"] == "scheduled" and datetime.fromisoformat(task["due_time"]) <= current_time:
                due_tasks.append(task)
        # Tasks are already sorted by _sort_schedule when added/updated, but re-sort for safety
        due_tasks.sort(key=lambda x: (datetime.fromisoformat(x['due_time']), -x['priority']))
        return due_tasks

    def _generate_next_recurrence(self, task: Dict[str, Any]) -> Optional[datetime]:
        """Generates the next due_time for a recurring task."""
        if not task.get("recurrence_pattern"):
            return None

        last_due_time = datetime.fromisoformat(task["due_time"])
        interval = task["recurrence_interval"]

        if task["recurrence_pattern"] == "daily":
            return last_due_time + timedelta(days=interval)
        elif task["recurrence_pattern"] == "weekly":
            return last_due_time + timedelta(weeks=interval)
        elif task["recurrence_pattern"] == "monthly":
            # Simple monthly recurrence: add interval months, keep day if possible
            year = last_due_time.year + (last_due_time.month + interval - 1) // 12
            month = (last_due_time.month + interval - 1) % 12 + 1
            day = min(last_due_time.day, (datetime(year, month + 1, 1) - timedelta(days=1)).day if month < 12 else 31) # Handle end of month
            return datetime(year, month, day, last_due_time.hour, last_due_time.minute, last_due_time.second, last_due_time.microsecond)
        return None

    def update_task(self, task_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Updates an existing task's properties.
        Can also be used to generate the next recurrence for a completed recurring task.
        """
        for i, task in enumerate(self.tasks):
            if task["task_id"] == task_id:
                # Update properties
                for key, value in kwargs.items():
                    if key in task: # Only update existing keys to prevent arbitrary additions
                        if key == "due_time" and not isinstance(value, str):
                            task[key] = value.isoformat()
                        else:
                            task[key] = value
                
                # If a recurring task is completed, schedule its next instance
                if task.get("status") == "completed" and task.get("recurrence_pattern"):
                    next_due_time = self._generate_next_recurrence(task)
                    if next_due_time:
                        # Create a new task for the next recurrence
                        new_task_details = {
                            "agent_id": task["agent_id"],
                            "description": task["description"],
                            "due_time": next_due_time,
                            "priority": task["priority"],
                            "recurrence_pattern": task["recurrence_pattern"],
                            "recurrence_interval": task["recurrence_interval"],
                            "metadata": task["metadata"]
                        }
                        new_task = self.schedule(new_task_details)
                        print(f"Scheduled next recurrence for task {task_id}: {new_task['task_id']} due {new_task['due_time']}")

                self._sort_schedule()
                self._save_schedule()
                return task
        return None

    def _set_task_status(self, task_id: str, status: Literal["scheduled", "pending", "completed", "cancelled", "failed"]) -> bool:
        """Helper to set task status and update timestamps."""
        for task in self.tasks:
            if task["task_id"] == task_id:
                task["status"] = status
                if status in ["completed", "failed", "cancelled"]:
                    task[f"{status}_at"] = datetime.now().isoformat()
                self._save_schedule()
                return True
        return False

    def mark_task_completed(self, task_id: str) -> bool:
        """Marks a specific task as completed."""
        return self._set_task_status(task_id, "completed")

    def mark_task_failed(self, task_id: str) -> bool:
        """Marks a specific task as failed."""
        return self._set_task_status(task_id, "failed")

    def mark_task_cancelled(self, task_id: str) -> bool:
        """Marks a specific task as cancelled."""
        return self._set_task_status(task_id, "cancelled")

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Returns all tasks in the schedule."""
        return self.tasks

    # --- CLI Integration ---
    def get_cli_commands(self) -> list:
        """
        Expose planner CLI command(s) to the host. Returns a list of mappings:
        { 'prefix': str, 'help': str, 'handler': callable(user_input, console, agent) }
        """
        help_text = (
            "help | list | schedule \"desc\" YYYY-MM-DD HH:MM [priority=<int>] "
            "[recurrence=daily|weekly|monthly] [interval=<int>] | complete <task_id> | cancel <task_id>"
        )
        return [{
            'prefix': ':plan',
            'help': help_text,
            'handler': self.handle_cli
        }]

    def handle_cli(self, user_input: str, console, agent) -> None:
        """Process a ':plan' command line and render output to console."""
        import shlex
        tokens = shlex.split(user_input)
        if len(tokens) < 2:
            console.print(self._help_line())
            return
        sub = tokens[1].lower()

        if sub in ("help", "h", "?"):
            console.print(self._help_line())
            return

        if sub == "list":
            for task in self.get_all_tasks():
                console.print(f"- {task['task_id']} | {task['description']} | due {task['due_time']} | prio {task['priority']} | status {task['status']}")
            return

        if sub == "complete":
            if len(tokens) < 3:
                console.print("[yellow]Usage: :plan complete <task_id>[/yellow]")
                return
            task_id = tokens[2]
            ok = self.mark_task_completed(task_id)
            console.print("[green]Marked completed.[/green]" if ok else "[red]Task not found.[/red]")
            return

        if sub == "cancel":
            if len(tokens) < 3:
                console.print("[yellow]Usage: :plan cancel <task_id>[/yellow]")
                return
            task_id = tokens[2]
            ok = self.mark_task_cancelled(task_id)
            console.print("[green]Cancelled.[/green]" if ok else "[red]Task not found.[/red]")
            return

        if sub == "schedule":
            if len(tokens) < 5:
                console.print("[yellow]Usage: :plan schedule \"desc\" YYYY-MM-DD HH:MM [priority=<int>] [recurrence=daily|weekly|monthly] [interval=<int>][/yellow]")
                return
            description = tokens[2]
            date_str = tokens[3]
            time_str = tokens[4]
            priority = 0
            recurrence = None
            interval = 1
            for opt in tokens[5:]:
                if opt.startswith("priority="):
                    try:
                        priority = int(opt.split("=", 1)[1])
                    except ValueError:
                        console.print("[yellow]priority must be an integer[/yellow]")
                        return
                elif opt.startswith("recurrence="):
                    recurrence = opt.split("=", 1)[1]
                elif opt.startswith("interval="):
                    try:
                        interval = int(opt.split("=", 1)[1])
                    except ValueError:
                        console.print("[yellow]interval must be an integer[/yellow]")
                        return

            from datetime import datetime
            try:
                due_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            except ValueError:
                console.print("[red]Invalid date/time. Use YYYY-MM-DD HH:MM[/red]")
                return

            details = {
                "agent_id": getattr(agent, "__class__", type(agent)).__name__,
                "description": description,
                "due_time": due_time,
                "priority": priority,
                "recurrence_pattern": recurrence,
                "recurrence_interval": interval,
                "metadata": {}
            }
            task = self.schedule(details)
            console.print(f"[green]Scheduled {task['task_id']} due {task['due_time']}[/green]")
            return

        # Fallback
        console.print(self._help_line())

    def _help_line(self) -> str:
        return (
            ":plan help | :plan list | :plan schedule \"desc\" YYYY-MM-DD HH:MM [priority=<int>] "
            "[recurrence=daily|weekly|monthly] [interval=<int>] | :plan complete <task_id> | :plan cancel <task_id>"
        )

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Note: The path is now relative to the new location
    test_planner_path = Path(__file__).parent / "planner_data.json"
    
    # Ensure the directory exists for the test
    test_planner_path.parent.mkdir(parents=True, exist_ok=True)

    planner = PlannerSystem(test_planner_path)

    # Clear previous test data
    if test_planner_path.exists():
        test_planner_path.unlink()
        planner = PlannerSystem(test_planner_path) # Re-initialize

    print("Scheduling tasks...")
    task1_details = {"agent_id": "gym_coach", "description": "Review workout log", "due_time": datetime.now() + timedelta(minutes=5), "priority": 1}
    task2_details = {"agent_id": "english_refinement", "description": "Proofread essay draft", "due_time": datetime.now() + timedelta(minutes=10), "priority": 2}
    task3_details = {"agent_id": "gym_coach", "description": "Plan next week's routine", "due_time": datetime.now() + timedelta(minutes=2), "priority": 3, "recurrence_pattern": "daily", "recurrence_interval": 1}
    task4_details = {"agent_id": "user_agent", "description": "Pay bills", "due_time": datetime.now() + timedelta(days=7), "priority": 5, "recurrence_pattern": "monthly", "recurrence_interval": 1}
    
    task1 = planner.schedule(task1_details)
    task2 = planner.schedule(task2_details)
    task3 = planner.schedule(task3_details)
    task4 = planner.schedule(task4_details)

    print("\nAll tasks after scheduling:")
    for task in planner.get_all_tasks():
        print(f"- {task['description']} (Due: {task['due_time']}, Priority: {task['priority']}, Status: {task['status']}, Recurrence: {task.get('recurrence_pattern')})")

    print("\nGetting upcoming tasks:")
    # We need to wait for some tasks to be due
    # For this test, we'll just get all tasks and pretend they are upcoming
    due_now = planner.getUpcomingTasks() 
    for task in due_now:
        print(f"- {task['description']} (Priority: {task['priority']})")
        if task['task_id'] == task3['task_id']:
             planner.mark_task_completed(task['task_id']) # Mark as completed, should trigger recurrence for task3

    print("\nAll tasks after marking some completed (check for recurrence):")
    for task in planner.get_all_tasks():
        print(f"- {task['description']} (Due: {task['due_time']}, Priority: {task['priority']}, Status: {task['status']}, Recurrence: {task.get('recurrence_pattern')})")

    # Test updating a task
    print("\nUpdating task2 description:")
    planner.update_task(task2['task_id'], description="Proofread final essay draft")
    for task in planner.get_all_tasks():
        if task['task_id'] == task2['task_id']:
            print(f"- {task['description']} (Due: {task['due_time']}, Priority: {task['priority']}, Status: {task['status']})")

    # Clean up test file
    if test_planner_path.exists():
        test_planner_path.unlink()
