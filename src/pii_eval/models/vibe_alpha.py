import re

from pii_eval.constants import (
    ISO_LABEL_DATE,
    ISO_LABEL_ID_CARD_NUMBER,
    ISO_LABEL_TAX_NUMBER,
    ISO_LABEL_URL,
    NON_ML_FEASIBLE_LABELS,
)
from pii_eval.data import PiiEntry, PiiModel
from pii_eval.models.regex_model import (
    REGEX_RULES,
    RegexRule,
    _compile,
    _iter_rule_spans,
    _select_non_overlapping_spans,
)
from pii_eval.span import LabelSpan


def _digits(value: str) -> str:
    return re.sub(r"\D+", "", value)


def _is_probable_url(value: str) -> bool:
    stripped = value.rstrip('.,;:!?)"]}>')
    if "." not in stripped:
        return False
    return bool(
        re.search(
            r"(?:https?://|www\.)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def _is_valid_numeric_date(value: str) -> bool:
    dmy = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2}|\d{4})", value)
    if dmy:
        day = int(dmy.group(1))
        month = int(dmy.group(2))
        return 1 <= day <= 31 and 1 <= month <= 12

    ymd = re.fullmatch(r"(19\d{2}|20\d{2})[./-](\d{1,2})[./-](\d{1,2})", value)
    if ymd:
        month = int(ymd.group(2))
        day = int(ymd.group(3))
        return 1 <= day <= 31 and 1 <= month <= 12

    return False


def _is_valid_tax_number(value: str) -> bool:
    compact = re.sub(r"[\s-]+", "", value).upper()
    digits = _digits(compact)
    if compact.startswith("DK"):
        return 8 <= len(digits) <= 12
    return 8 <= len(digits) <= 14


def _is_valid_id_card_number(value: str) -> bool:
    compact = re.sub(r"[\s-]+", "", value).upper()
    if not compact:
        return False
    if compact.startswith("DK"):
        compact = compact[2:]
    digits = _digits(compact)
    return compact.isalnum() and 7 <= len(digits) <= 14


def _align_to_entry_tokens(
    tokens: list[LabelSpan], spans: list[LabelSpan]
) -> list[LabelSpan]:
    started_spans: set[int] = set()
    aligned: list[LabelSpan] = []

    for token in tokens:
        label = "O"
        for i, span in enumerate(spans):
            if token.start >= span.start and token.end <= span.end:
                prefix = "I" if i in started_spans else "B"
                started_spans.add(i)
                label = f"{prefix}-{span.label}"
                break
        aligned.append(
            LabelSpan(
                start=token.start,
                end=token.end,
                text=token.text,
                label=label,
            )
        )

    return aligned


VIBE_ALPHA_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        label=ISO_LABEL_URL,
        pattern=_compile(r"(?P<value>\b(?:https?://|www\.)[^\s<>'\"\\)\]\}]+)"),
        validator=_is_probable_url,
        priority=120,
    ),
    RegexRule(
        label=ISO_LABEL_DATE,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:\d{2}|\d{4})\b
            )
            """
        ),
        validator=_is_valid_numeric_date,
        priority=130,
    ),
    RegexRule(
        label=ISO_LABEL_DATE,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:19\d{2}|20\d{2})[./-](?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12]\d|3[01])\b
            )
            """
        ),
        validator=_is_valid_numeric_date,
        priority=131,
    ),
    RegexRule(
        label=ISO_LABEL_DATE,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:
                    jan(?:uary|uar)?
                    |feb(?:ruary|ruar)?
                    |mar(?:ch|ts)?
                    |apr(?:il)?
                    |maj
                    |may
                    |jun(?:e|i)?
                    |jul(?:y|i)?
                    |aug(?:ust)?
                    |sep(?:t(?:ember)?)?
                    |okt(?:ober)?
                    |oct(?:ober)?
                    |nov(?:ember)?
                    |dec(?:ember)?
                    |des(?:ember)?
                )\s+(?:19\d{2}|20\d{2})
                \b
            )
            """
        ),
        priority=132,
    ),
    RegexRule(
        label=ISO_LABEL_DATE,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:
                    i\s+dag
                    |i\s+morgen
                    |today
                    |tomorrow
                    |yesterday
                    |mandag
                    |tirsdag
                    |onsdag
                    |torsdag
                    |fredag
                    |monday
                    |tuesday
                    |wednesday
                    |thursday
                    |friday
                    |saturday
                    |sunday
                    |jan(?:uary|uar)?
                    |feb(?:ruary|ruar)?
                    |mar(?:ch|ts)?
                    |apr(?:il)?
                    |maj
                    |may
                    |jun(?:e|i)?
                    |jul(?:y|i)?
                    |aug(?:ust)?
                    |sep(?:t(?:ember)?)?
                    |okt(?:ober)?
                    |oct(?:ober)?
                    |nov(?:ember)?
                    |dec(?:ember)?
                    |des(?:ember)?
                )\b
            )
            """
        ),
        priority=133,
    ),
    RegexRule(
        label=ISO_LABEL_DATE,
        pattern=_compile(r"(?P<value>\b(?:19\d{2}|20\d{2})\b)"),
        priority=134,
    ),
    RegexRule(
        label=ISO_LABEL_TAX_NUMBER,
        pattern=_compile(
            r"""
            \b(?:
                tax(?:\s*(?:id|identifier|number|no\.?))?
                |taxnum
                |tin
                |vat(?:\s*(?:id|number|no\.?))?
                |cvr(?:[\-\s]?(?:nr\.?|nummer))?
                |se(?:[\-\s]?nr\.?)
                |moms(?:[\-\s]?nummer)?
                |skatte?(?:[\-\s]?(?:id|nummer|identifikationsnummer|personnummer|kortnummer))?
            )\b
            [^\n\r\d]{0,30}?
            (?P<value>
                (?:DK[- ]?)?\d{6}[- ]?\d{4}
                |
                (?:DK[- ]?)?\d{8,12}
                |
                \d{3,4}(?:[- ]\d{3,4}){2,3}
            )
            """
        ),
        validator=_is_valid_tax_number,
        priority=44,
    ),
    RegexRule(
        label=ISO_LABEL_TAX_NUMBER,
        pattern=_compile(
            r"""
            \b(?:tax(?:\s*(?:id|number))?|skatte?(?:id|nummer)|cvr(?:[\-\s]?nr\.?)?|moms(?:nummer)?)\b
            [^\n\r]{0,45}?
            (?P<value>(?:DK[- ]?)?\d{8,12}|\d{6}[- ]\d{4})
            """
        ),
        validator=_is_valid_tax_number,
        priority=45,
    ),
    RegexRule(
        label=ISO_LABEL_TAX_NUMBER,
        pattern=_compile(r"(?P<value>\bDK[- ]?\d{8,12}\b)"),
        validator=_is_valid_tax_number,
        priority=46,
    ),
    RegexRule(
        label=ISO_LABEL_ID_CARD_NUMBER,
        pattern=_compile(
            r"""
            \b(?:
                id(?:[\-\s\u2010-\u2015]?(?:card|kort|nummer|nr\.?|no\.?))?
                |id[\-\s\u2010-\u2015]?kort(?:nummer)?
                |identity\s*card
                |national(?:t)?[\-\s\u2010-\u2015]?id(?:[\-\s\u2010-\u2015]?nummer)?
                |identitetskort(?:nummer)?
                |identifikationsnummer
                |legitimation(?:snummer)?
            )\b
            [^\n\r]{0,45}?
            (?P<value>
                (?:DK[- ]?)?[A-Z0-9](?:[A-Z0-9 -]{5,22}[A-Z0-9])?
            )
            """
        ),
        validator=_is_valid_id_card_number,
        priority=43,
    ),
    RegexRule(
        label=ISO_LABEL_ID_CARD_NUMBER,
        pattern=_compile(
            r"""
            \b(?:id|id\s*card|id\s*kort)\b\s*[:#=\-\u2010-\u2015]\s*(?P<value>(?:DK[- ]?)?[A-Z0-9](?:[A-Z0-9 -]{5,18}[A-Z0-9])?)
            """
        ),
        validator=_is_valid_id_card_number,
        priority=42,
    ),
    RegexRule(
        label=ISO_LABEL_ID_CARD_NUMBER,
        pattern=_compile(r"(?P<value>\bDK[- ]?\d{8,12}\b)"),
        validator=_is_valid_id_card_number,
        priority=47,
    ),
)


class VibeAlpha(PiiModel):
    def __init__(
        self,
        name: str = "vibe-alpha",
    ):
        super().__init__(name)

    def __call__(self, entry: PiiEntry) -> PiiEntry:
        active_labels = NON_ML_FEASIBLE_LABELS

        prioritized_spans: list[tuple[int, LabelSpan]] = []
        for rule in (*REGEX_RULES, *VIBE_ALPHA_RULES):
            if rule.label not in active_labels:
                continue
            for span in _iter_rule_spans(entry.text, rule):
                prioritized_spans.append((rule.priority, span))

        predicted_spans = _select_non_overlapping_spans(prioritized_spans)

        predicted_bio = _align_to_entry_tokens(entry.bio, predicted_spans)
        return PiiEntry(
            uid=entry.uid,
            text=entry.text,
            bio=predicted_bio,
            meta=entry.meta,
        )
