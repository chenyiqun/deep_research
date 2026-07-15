"""Compatibility import for the event-driven multi-agent workflow.

The implementation lives in :mod:`drb_qwen.multi_agent`. Keeping this module
preserves the original CLI and evaluation imports.
"""

from .multi_agent.workflow import AsyncDeepResearchWorkflow, DeepResearchConfig

__all__ = ["AsyncDeepResearchWorkflow", "DeepResearchConfig"]
