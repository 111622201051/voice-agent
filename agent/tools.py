# agent/tools.py
# Lightweight tool registry so the agent can expose callable "skills"
# (clock lookups, system status, etc.) to the LLM without external deps.
import datetime
import logging

logger = logging.getLogger(__name__)


class Tool:
    def __init__(self, name: str, description: str, handler):
        self.name = name
        self.description = description
        self.handler = handler

    def __call__(self, **kwargs):
        return self.handler(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
        return tool

    def register_handler(self, name: str, description: str, handler):
        return self.register(Tool(name, description, handler))

    def get(self, name: str):
        return self._tools.get(name)

    def call(self, name: str, **kwargs):
        tool = self.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        try:
            return tool(**kwargs)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return f"Tool '{name}' error: {e}"

    def all_tools(self):
        return {name: tool.description for name, tool in self._tools.items()}


# --- Built-in tools ---------------------------------------------------------
def _get_current_time(**kwargs):
    return datetime.datetime.now().strftime("%I:%M %p")


def _get_current_date(**kwargs):
    return datetime.datetime.now().strftime("%A, %B %d, %Y")


def _get_session_timestamp(**kwargs):
    return datetime.datetime.now().isoformat(timespec="seconds")


def build_default_registry() -> ToolRegistry:
    """Registry with the standard built-in tools used by the agent."""
    reg = ToolRegistry()
    reg.register_handler("get_current_time",
                         "Get the current local time.",
                         _get_current_time)
    reg.register_handler("get_current_date",
                         "Get today's full date and day name.",
                         _get_current_date)
    reg.register_handler("session_timestamp",
                         "Get the current UTC timestamp for the session log.",
                         _get_session_timestamp)
    return reg


default_registry = build_default_registry()