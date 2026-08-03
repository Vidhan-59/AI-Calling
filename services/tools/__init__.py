"""Importing this package registers every tool (the @tool decorator runs on
import) - anything that needs invoke_tool/get_tool/list_tools to see the
full catalog should `import services.tools` first."""

from services.tools import comms_tools, hr_tools, roster_tools, transfer_tools  # noqa: F401
from services.tools.registry import get_tool, invoke_tool, list_tools, tool  # noqa: F401
