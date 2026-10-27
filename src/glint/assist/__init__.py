"""Decision tools that assist Glimmer: deterministic answers to the analytical questions in τ² tasks."""


def as_flag(value) -> bool:
    """A boolean tool argument as the model wrote it. τ² shows every bool to the model as "True"/"False", so a model
    can send the string back, and bool("False") is True (2026-10-27 review: a "False" leaked the withheld airline
    workaround hint)."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)
