from typing import Optional
from rich.console import Console


class AgentHost:
	"""Thin runtime wrapper for a single agent instance."""
	def __init__(self, agent, console: Optional[Console] = None):
		self.agent = agent
		self.console = console or Console()

	def start(self):
		"""Start the agent runtime (lifecycle hook)."""
		if hasattr(self.agent, "on_start"):
			self.agent.on_start()

	def stop(self):
		"""Stop the agent runtime (lifecycle hook)."""
		if hasattr(self.agent, "on_stop"):
			self.agent.on_stop()

	def run_cli(self, prompt_label: str = "agent"):
		"""Run a simple blocking CLI loop supporting journal input and system-defined commands."""
		self.console.print("✅ CLI ready. Type your journal entry. Type 'exit' to quit.")
		# Collect CLI commands from systems that expose get_cli_commands()
		commands = []
		context = getattr(self.agent, "context", object())
		for attr in dir(context):
			if attr.startswith("_"):
				continue
			sys_obj = getattr(context, attr, None)
			if sys_obj and hasattr(sys_obj, "get_cli_commands") and callable(getattr(sys_obj, "get_cli_commands")):
				try:
					for cmd in sys_obj.get_cli_commands() or []:
						if isinstance(cmd, dict) and "prefix" in cmd and "handler" in cmd:
							commands.append(cmd)
				except Exception:
					pass
		# Render help for each command provider
		for cmd in commands:
			help_line = cmd.get("help")
			if help_line:
				self.console.print(f"   {cmd['prefix']} {help_line}")
		while True:
			try:
				user_input = self.console.input(f"\n[bold green]{prompt_label}>[/bold green] ").strip()
				if not user_input:
					continue
				if user_input.lower() == "exit":
					break

				# System command dispatch
				matched = False
				for cmd in commands:
					prefix = cmd.get("prefix")
					handler = cmd.get("handler")
					if isinstance(prefix, str) and user_input.startswith(prefix) and callable(handler):
						try:
							handler(user_input, self.console, self.agent)
							matched = True
							break
						except Exception as e:
							self.console.print(f"[bold red]❌ Command error: {e}[/bold red]")
							matched = True
							break
				if matched:
					continue

				# Otherwise, treat as a journal entry
				self.agent.save_entry(user_input)
				self.console.print("[bold green]Journal entry added.[/bold green]")
			except KeyboardInterrupt:
				break
			except Exception as e:
				self.console.print(f"[bold red]❌ Error: {e}[/bold red]")

	def _handle_plan_command(self, user_input: str) -> None:
		"""Parse and execute :plan commands routed to the agent's planner_system."""
		planner = getattr(getattr(self.agent, "context", object()), "planner_system", None)
		if planner is None:
			self.console.print("[bold red]Planner system is not available for this agent.[/bold red]")
			return

		# Tokenize while preserving quoted description
		import shlex
		tokens = shlex.split(user_input)
		# tokens example: [":plan", "schedule", "desc", "2025-10-15", "09:00", "priority=5", "recurrence=monthly", "interval=1"]
		if len(tokens) < 2:
			self._print_plan_help()
			return
		sub = tokens[1].lower()

		if sub in ("help", "h", "?"):
			self._print_plan_help()
			return

		if sub == "list":
			for task in planner.get_all_tasks():
				self.console.print(f"- {task['task_id']} | {task['description']} | due {task['due_time']} | prio {task['priority']} | status {task['status']}")
			return

		if sub == "complete":
			if len(tokens) < 3:
				self.console.print("[yellow]Usage: :plan complete <task_id>[/yellow]")
				return
			task_id = tokens[2]
			ok = planner.mark_task_completed(task_id)
			self.console.print("[green]Marked completed.[/green]" if ok else "[red]Task not found.[/red]")
			return

		if sub == "cancel":
			if len(tokens) < 3:
				self.console.print("[yellow]Usage: :plan cancel <task_id>[/yellow]")
				return
			task_id = tokens[2]
			ok = planner.mark_task_cancelled(task_id)
			self.console.print("[green]Cancelled.[/green]" if ok else "[red]Task not found.[/red]")
			return

		if sub == "schedule":
			if len(tokens) < 5:
				self.console.print("[yellow]Usage: :plan schedule \"desc\" YYYY-MM-DD HH:MM [priority=<int>] [recurrence=daily|weekly|monthly] [interval=<int>][/yellow]")
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
						self.console.print("[yellow]priority must be an integer[/yellow]")
						return
				elif opt.startswith("recurrence="):
					recurrence = opt.split("=", 1)[1]
				elif opt.startswith("interval="):
					try:
						interval = int(opt.split("=", 1)[1])
					except ValueError:
						self.console.print("[yellow]interval must be an integer[/yellow]")
						return

			from datetime import datetime
			try:
				due_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
			except ValueError:
				self.console.print("[red]Invalid date/time. Use YYYY-MM-DD HH:MM[/red]")
				return

			details = {
				"agent_id": getattr(self.agent, "__class__", type(self.agent)).__name__,
				"description": description,
				"due_time": due_time,
				"priority": priority,
				"recurrence_pattern": recurrence,
				"recurrence_interval": interval,
				"metadata": {}
			}
			task = planner.schedule(details)
			self.console.print(f"[green]Scheduled {task['task_id']} due {task['due_time']}[/green]")
			return

		self._print_plan_help()

	def _print_plan_help(self) -> None:
		self.console.print(
			":plan help | :plan list | :plan schedule \"desc\" YYYY-MM-DD HH:MM [priority=<int>] [recurrence=daily|weekly|monthly] [interval=<int>] | :plan complete <task_id> | :plan cancel <task_id>"
		)


