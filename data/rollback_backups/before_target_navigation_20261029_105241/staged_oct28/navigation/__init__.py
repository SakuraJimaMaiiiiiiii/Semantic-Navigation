"""Model-independent semantic navigation and target selection interfaces."""

from .target import SemanticTargetRequest, load_target_request, select_targets

__all__ = ["SemanticTargetRequest", "load_target_request", "select_targets"]
