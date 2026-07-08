from typing import Protocol

from pii_eval.data import PiiModel
from pii_eval.models.regex_model import Regex


class Replacer(Protocol):
    def __call__(self, label: str, substring: str) -> str: ...


def default_replacer(label: str, _substring: str) -> str:
    return f"<{label}>"


def redact(
    text: str,
    model: PiiModel = Regex(),
    replacer: Replacer = default_replacer,
) -> str:
    spans = model.predict_spans(text=text)
    if not spans:
        return text
    last_idx = 0
    substrs: list[str] = []
    for span in spans:
        substrs.append(text[last_idx : span.start])
        substrs.append(replacer(span.label, span.text))
        last_idx = span.end
    substrs.append(text[last_idx:])
    return "".join(substrs)


if __name__ == "__main__":
    print(redact("Hello my phone number is +45 12 23 34 45."))
