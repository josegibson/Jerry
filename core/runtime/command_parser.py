from typing import Tuple, Optional, List


class CommandParser:
    """
    Parses user commands for agent interaction.
    Handles both agent.method(args) and method(args) formats.
    """
    
    @staticmethod
    def parse_command(user_input: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Parse a user command string.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Tuple of (agent_name, method_name, args)
            - agent_name: None if no agent specified, otherwise agent name
            - method_name: The method to call
            - args: List of arguments (currently supports single string arg)
        """
        try:
            command_part, args_part = user_input.split('(', 1)
            if not args_part.endswith(')'):
                return None, None, []
            
            arg_str = args_part[:-1]
            
            if '.' in command_part:
                agent_name, method_name = command_part.split('.', 1)
            else:
                agent_name, method_name = None, command_part
            
            args = []
            if arg_str:
                # Simple parsing handles a single string argument
                # Can be improved to handle multiple args, numbers, etc.
                if arg_str.startswith('"') and arg_str.endswith('"'):
                    args.append(arg_str[1:-1])
                else:
                    args.append(arg_str)  # Treat as a single value
            
            return agent_name, method_name, args
            
        except ValueError:
            return None, None, []
    
    @staticmethod
    def is_valid_command_format(user_input: str) -> bool:
        """
        Check if the input is in a valid command format.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            True if valid command format, False otherwise
        """
        agent_name, method_name, args = CommandParser.parse_command(user_input)
        return method_name is not None

