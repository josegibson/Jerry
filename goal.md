The Life Canvas: A Design Charter
1. Core Principle: The Vault is the Ground Truth
The system is built on a single, foundational principle: The user's Obsidian vault is the absolute ground truth. It is not merely a data source; it is the world, the system state, and the canvas upon which a user's life is painted. The agents are assistants who operate within this world. Their understanding is derived directly from the human-readable Markdown files and folder structures of the vault. This ensures the system is durable, transparent, and always under the user's ultimate control.

2. The Core Unit: The Initiative
The fundamental unit of work is the Initiative, which represents any focused effort and is always embodied by a folder in the vault. Each Initiative is managed by its own agent, which acts as a specialized assistant for that effort.

The Initiative Folder Structure
The structure is designed to be human-first, separating user-authored notes from system configurations.

📁 My Initiative/
│
├── 📄 roadmap.md          (✅ Human-Authored: The strategic plan with phases/milestones)
├── 📄 tasks.md            (✅ Human-Authored: The tactical to-do list with a defined syntax)
├── 📄 entries.md           (✅ Human-Authored: The inbox for raw thoughts and session logs)
│
├── 📄 _initiative.yaml     (⚠️ UI-Managed: The agent's "control panel" for configs and state)
│
└── 📁 sub-initiatives/     (The home for nested child Initiatives)

The Initiative State (_initiative.yaml)
This UI-managed file is the agent's state manager. It contains:

id and displayName.

priority_to_parent: A numerical weight (0.0-1.0) declaring its importance to its parent, enabling the recursive priority system.

current_stage: The Initiative's current phase, read from roadmap.md and updated by the agent upon phase completion.

scoring_providers: A list of pluggable scoring modules and their configurations (e.g., user_defined_affinity, momentum_pressure) that dictate how tasks are prioritized.

activity_log.jsonl: An append-only, machine-readable record of all significant events (tasks completed, stages changed), providing a historical context for intelligent score providers.

3. The Planner: A Stateless, Recursive Compiler
The Planner is not a stateful scheduler; it is a stateless compiler. Its sole job is to read the ground truth from the vault at any given moment and generate a dynamic, prioritized "Today's Plan" in memory.

The Recursive Priority Roll-Up
The Planner calculates a Global Score for every task in the hierarchy using a recursive algorithm.

Global Score = Local Priority * Inherited Priority * (Product of all Scoring Provider Multipliers)

Local Priority: A number (e.g., 1-10) defined directly in tasks.md (e.g., p:9).

Inherited Priority: A value calculated by multiplying the priority_to_parent of every Initiative down the hierarchy.

Scoring Providers: A modular, pluggable system that allows for new, intelligent scoring methods to be added over time. The Planner calls each provider listed in the _initiative.yaml to get its multiplier (e.g., for time affinity, procrastination pressure, or goal urgency).

This ensures the "Today's Plan" is always a real-time, mathematically sound reflection of the user's stated priorities at every level of their life.

4. The User Experience: The Vault and The App
The user interacts with the system through two complementary interfaces.

The Vault (Direct Manipulation): The user has ultimate control by directly editing the human-readable .md files in Obsidian.

The Application (Guided Interaction): A UI (CLI or GUI) that acts as an intelligent lens on the vault.

Dashboard: Provides the dynamic "Today's Plan" for the current Initiative context.

Breadcrumb Navigation: Allows for frictionless movement up and down the Initiative hierarchy.

Universal Inbox: A global journal for capturing thoughts that can be semantically routed to the correct Initiative.

Quick Capture: A universal command (e.g., :plan) for creating miscellaneous tasks that default to the root "Life" Initiative.

5. The Awareness Tool: Version Control
To transform the system into a tool for self-awareness, the entire vault is a single Git repository.

The Agent as a "Git Co-pilot": The system differentiates between routine work and major decisions.

Strategic Commits: The user is prompted for a reflective commit message only when making a significant, plan-altering change (e.g., putting an Initiative on hold).

Daily Roll-Up Commits: Routine progress (tasks completed, entries added) is bundled into clean, auto-generated summary commits at the end of the day.

The "Done... Now What?" Protocol: Upon task completion, the system offers an optional prompt for a journal reflection, creating a powerful, auditable link between action and thought.

This creates a high-signal, annotated history of the user's intentions, decisions, and progress, fulfilling the ultimate vision of a "Life Canvas."