from typing import Any, Callable

from pii_eval.span import LabelSpan

type Replacer = Callable[[str, str, Any | None], str]


def default_replacer(label: str, _substring: str, _state: Any | None = None) -> str:
    return f"<{label}>"


def redact(
    self,
    text: str,
    model: Callable[[str], list[LabelSpan]],
    replacer: Replacer = default_replacer,
) -> str:
    entities = self.model(text)
    if len(entities) == 0:
        return text
    substrs = []
    state = None
    last_idx = 0
    for entity in entities:
        substrs.append(text[last_idx : entities[0].start])
        substrs.append(
            entity.text
            if entity.label == "O"
            else replacer(entity.label, entity.text, state)
        )
        last_idx = entity.end
    substrs.append(text[last_idx:])
    return "".join(substrs)
