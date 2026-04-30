"""News triage agent package."""

__all__ = ["run_triage"]


def __getattr__(name: str):
    if name == "run_triage":
        from .workflow import run_triage

        return run_triage
    raise AttributeError(name)
