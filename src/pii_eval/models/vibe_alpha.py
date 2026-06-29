import re
from collections import defaultdict
from datetime import datetime

from pii_eval.constants import (
    ISO_LABEL_BANK_ACCOUNT,
    ISO_LABEL_DATE,
    ISO_LABEL_DRIVERS_LICENSE,
    ISO_LABEL_EMAIL_ADDRESS,
    ISO_LABEL_ID_CARD_NUMBER,
    ISO_LABEL_IP_ADDRESS,
    ISO_LABEL_PASSPORT,
    ISO_LABEL_PCI,
    ISO_LABEL_PHONE_NUMBER,
    ISO_LABEL_SSN,
    ISO_LABEL_TAX_NUMBER,
    ISO_LABEL_URL,
    ISO_LABEL_VEHICLE_ID,
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


def _compact(value: str) -> str:
    return re.sub(r"[\s-]+", "", value)


def _is_probable_url(value: str) -> bool:
    stripped = value.rstrip('.,;:!?)"]}>')
    lowered = stripped.lower()
    if "linkedin.com/in/" in lowered:
        return False
    if ".local/" in lowered or lowered.endswith(".local"):
        return False
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
    compact = _compact(value).upper()
    normalized = compact.removeprefix("DK").removeprefix("DS").removesuffix("DK")
    digits = _digits(normalized)
    if compact.startswith("DK") or compact.startswith("DS"):
        return 8 <= len(digits) <= 12
    return 8 <= len(digits) <= 14


def _is_valid_id_card_number(value: str) -> bool:
    compact = _compact(value).upper()
    if not compact:
        return False
    normalized = compact.removeprefix("DK").removeprefix("DS")
    digits = _digits(normalized)
    return normalized.isalnum() and 7 <= len(digits) <= 14


def _looks_like_drivers_license(value: str) -> bool:
    compact = _compact(value).upper()
    digits = _digits(compact)
    letter_count = sum(ch.isalpha() for ch in compact)
    return (
        compact.isalnum()
        and 5 <= len(compact) <= 20
        and (letter_count >= 1 or len(digits) >= 8)
    )


def _has_valid_ddmmyy_prefix(value: str) -> bool:
    compact = _digits(value)
    if len(compact) < 6:
        return False

    day = int(compact[:2])
    month = int(compact[2:4])
    year_suffix = int(compact[4:6])
    for century in (1900, 2000):
        try:
            datetime(century + year_suffix, month, day)
            return True
        except ValueError:
            continue
    return False


def _looks_like_ssn(value: str) -> bool:
    stripped = value.strip().upper()
    if re.fullmatch(r"\d{6}[- ]?\d{4}", stripped):
        return _has_valid_ddmmyy_prefix(stripped)
    if re.fullmatch(r"\d{6}[- ]?\d{3}[A-Z]", stripped):
        return _has_valid_ddmmyy_prefix(stripped)

    compact_digits = _digits(stripped)
    if len(compact_digits) == 10:
        return _has_valid_ddmmyy_prefix(compact_digits)

    return False


def _looks_like_vehicle_id(value: str) -> bool:
    compact = _compact(value).upper()
    return bool(
        re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", compact)
        or re.fullmatch(r"[A-Z]{1,3}\d{4,6}", compact)
        or re.fullmatch(r"[A-Z]{2}\d{2}\d{3}", compact)
        or re.fullmatch(r"[A-Z]{2}\d{3}\d{2}", compact)
    )


def _looks_like_passport(value: str) -> bool:
    compact = _compact(value).upper()
    if "<" in value:
        return True
    digit_count = len(_digits(compact))
    letter_count = sum(ch.isalpha() for ch in compact)
    return (
        compact.isalnum()
        and 6 <= len(compact) <= 12
        and (letter_count >= 1 or 8 <= digit_count <= 10)
    )


def _looks_like_id_card(value: str) -> bool:
    compact = _compact(value).upper()
    digit_count = len(_digits(compact))
    return compact.isalnum() and 8 <= digit_count <= 14


def _looks_like_phone_number_value(value: str) -> bool:
    stripped = value.strip()
    digits = _digits(stripped)
    if not (8 <= len(digits) <= 15):
        return False
    if re.fullmatch(r"\d{4}-\d{4}", stripped):
        return False
    if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", stripped):
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return False
    if "+" in stripped:
        return True
    separator_count = sum(ch in " .()-" for ch in stripped)
    return separator_count >= 2 or len(digits) == 8


def _context_window(text: str, start: int, end: int, size: int = 96) -> str:
    left = max(0, start - size)
    right = min(len(text), end + size)
    return text[left:right].lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_probable_email(value: str) -> bool:
    stripped = value.rstrip('.,;:!?)"]}>')
    if stripped.count("@") != 1:
        return False
    local, domain = stripped.split("@", 1)
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    if " " in local or " " in domain:
        return False
    return True


def _normalize_context_text(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(
        r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", normalized
    )
    return normalized


def _nearest_keyword_distance(
    text: str, start: int, keywords: tuple[str, ...], max_distance: int = 64
) -> int | None:
    left = max(0, start - max_distance)
    snippet = _normalize_context_text(text[left:start])
    best: int | None = None

    for keyword in keywords:
        normalized_keyword = _normalize_context_text(keyword)
        idx = snippet.rfind(normalized_keyword)
        if idx < 0:
            continue
        distance = len(snippet) - (idx + len(normalized_keyword))
        if best is None or distance < best:
            best = distance

    return best


TAX_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "tax",
    "taxnum",
    "tax id",
    "tax number",
    "tin",
    "vat",
    "skat",
    "skatte",
    "skattem",
    "skattepersonnummer",
    "skattekortnummer",
    "skatte-id",
    "cvr",
    "moms",
    "se-nr",
)

ID_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "id card",
    "id-card",
    "idkort",
    "id-kort",
    "identity card",
    "identitetskort",
    "identifikationsnummer",
    "legitimation",
)

DRIVERS_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "driver",
    "license",
    "licence",
    "driving",
    "k\u00f8rekort",
    "korekort",
    "f\u00f8rerkort",
)

PHONE_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "phone",
    "telefon",
    "telefonnummer",
    "mobil",
    "mobilnummer",
    "tel",
    "tlf",
)

SSN_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "cpr",
    "ssn",
    "social security",
    "socialsikr",
    "personnummer",
    "national id",
    "national-id",
    "nationalt id",
    "nationale id",
    "folkeregistreringsnummer",
    "civil registration",
)

PASSPORT_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "passport",
    "passnummer",
    "pasnummer",
    "pas nr",
    "pasnr",
    "rejsepas",
)

VEHICLE_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "vehicle",
    "license plate",
    "number plate",
    "vin",
    "chassis",
    "nummerplade",
    "registreringsnummer",
    "regnr",
    "reg nr",
    "stelnummer",
)


def _apply_context_relabeling(text: str, spans: list[LabelSpan]) -> list[LabelSpan]:
    ambiguous_labels = {
        ISO_LABEL_PHONE_NUMBER,
        ISO_LABEL_SSN,
        ISO_LABEL_TAX_NUMBER,
        ISO_LABEL_ID_CARD_NUMBER,
        ISO_LABEL_PASSPORT,
        ISO_LABEL_DRIVERS_LICENSE,
        ISO_LABEL_VEHICLE_ID,
    }

    relabeled: list[LabelSpan] = []
    for span in spans:
        label = span.label
        compact_text = _compact(span.text).upper()
        digit_count = len(_digits(compact_text))
        letter_count = sum(ch.isalpha() for ch in compact_text)
        is_digits_only = bool(compact_text) and compact_text.isdigit()
        phone_like = _looks_like_phone_number_value(span.text)

        looks_like_ssn = _looks_like_ssn(span.text)
        looks_like_passport = _looks_like_passport(span.text)
        looks_like_id_card = _looks_like_id_card(span.text)
        looks_like_driver = _looks_like_drivers_license(span.text)
        looks_like_vehicle = _looks_like_vehicle_id(span.text)
        looks_like_tax = _is_valid_tax_number(span.text)
        ssn_distance = _nearest_keyword_distance(text, span.start, SSN_CONTEXT_KEYWORDS)
        tax_distance = _nearest_keyword_distance(text, span.start, TAX_CONTEXT_KEYWORDS)
        id_distance = _nearest_keyword_distance(text, span.start, ID_CONTEXT_KEYWORDS)
        driver_distance = _nearest_keyword_distance(
            text, span.start, DRIVERS_CONTEXT_KEYWORDS
        )
        phone_distance = _nearest_keyword_distance(
            text, span.start, PHONE_CONTEXT_KEYWORDS
        )
        passport_distance = _nearest_keyword_distance(
            text, span.start, PASSPORT_CONTEXT_KEYWORDS
        )
        vehicle_distance = _nearest_keyword_distance(
            text, span.start, VEHICLE_CONTEXT_KEYWORDS
        )

        if label in ambiguous_labels:
            candidates: list[tuple[int, int, str]] = []

            def _add_candidate(distance: int, rank: int, candidate_label: str) -> None:
                adjusted_distance = distance
                if candidate_label == span.label:
                    adjusted_distance -= 4
                if candidate_label == ISO_LABEL_PASSPORT and is_digits_only:
                    adjusted_distance += 6
                candidates.append((adjusted_distance, rank, candidate_label))

            if (
                looks_like_vehicle
                and vehicle_distance is not None
                and vehicle_distance <= 24
            ):
                _add_candidate(vehicle_distance, 0, ISO_LABEL_VEHICLE_ID)

            if (
                looks_like_ssn
                and looks_like_tax
                and tax_distance is not None
                and tax_distance <= 20
                and (ssn_distance is None or tax_distance <= ssn_distance + 2)
            ):
                _add_candidate(max(tax_distance - 2, 0), 1, ISO_LABEL_TAX_NUMBER)

            if looks_like_ssn and ssn_distance is not None and ssn_distance <= 28:
                if not (
                    tax_distance is not None
                    and tax_distance + 2 < ssn_distance
                    and label == ISO_LABEL_TAX_NUMBER
                ):
                    _add_candidate(ssn_distance, 2, ISO_LABEL_SSN)

            if (
                looks_like_tax
                and tax_distance is not None
                and tax_distance <= 34
                and (ssn_distance is None or tax_distance <= ssn_distance + 3)
            ):
                _add_candidate(tax_distance, 3, ISO_LABEL_TAX_NUMBER)

            if (
                looks_like_id_card
                and not looks_like_ssn
                and id_distance is not None
                and id_distance <= 32
            ):
                _add_candidate(id_distance, 4, ISO_LABEL_ID_CARD_NUMBER)

            if (
                looks_like_driver
                and driver_distance is not None
                and driver_distance <= 28
            ):
                _add_candidate(driver_distance, 5, ISO_LABEL_DRIVERS_LICENSE)

            if (
                looks_like_passport
                and not looks_like_ssn
                and passport_distance is not None
                and passport_distance <= 22
                and (
                    "<" in span.text
                    or letter_count > 0
                    or passport_distance <= 12
                    or (
                        is_digits_only
                        and 8 <= digit_count <= 9
                        and passport_distance <= 20
                    )
                )
            ):
                _add_candidate(passport_distance, 6, ISO_LABEL_PASSPORT)

            if (
                phone_like
                and phone_distance is not None
                and phone_distance <= 18
                and ("+" in span.text or not looks_like_ssn)
            ):
                _add_candidate(phone_distance, 7, ISO_LABEL_PHONE_NUMBER)

            if candidates:
                label = min(candidates)[2]

            if (
                label == ISO_LABEL_PASSPORT
                and is_digits_only
                and 8 <= digit_count <= 10
                and passport_distance is None
            ):
                if id_distance is not None and id_distance <= 34:
                    label = ISO_LABEL_ID_CARD_NUMBER
                elif tax_distance is not None and tax_distance <= 24:
                    label = ISO_LABEL_TAX_NUMBER
                elif ssn_distance is not None and ssn_distance <= 24 and looks_like_ssn:
                    label = ISO_LABEL_SSN

            if label == ISO_LABEL_PHONE_NUMBER:
                if "<" in span.text and len(compact_text) >= 16:
                    label = ISO_LABEL_PASSPORT
                elif (
                    is_digits_only
                    and 8 <= digit_count <= 9
                    and passport_distance is not None
                    and passport_distance <= 18
                ):
                    label = ISO_LABEL_PASSPORT
                elif not phone_like:
                    if (
                        looks_like_ssn
                        and ssn_distance is not None
                        and ssn_distance <= 28
                    ):
                        label = ISO_LABEL_SSN
                    elif (
                        looks_like_tax
                        and tax_distance is not None
                        and tax_distance <= 24
                    ):
                        label = ISO_LABEL_TAX_NUMBER
                    elif (
                        looks_like_id_card
                        and not looks_like_ssn
                        and id_distance is not None
                        and id_distance <= 30
                    ):
                        label = ISO_LABEL_ID_CARD_NUMBER

        if label == span.label:
            relabeled.append(span)
        else:
            relabeled.append(
                LabelSpan(
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    label=label,
                )
            )
    return relabeled


def _align_to_entry_tokens(
    tokens: list[LabelSpan], spans: list[LabelSpan]
) -> list[LabelSpan]:
    sorted_spans = sorted(spans, key=lambda span: (span.start, span.end))
    span_index = 0
    active_span_index = -1
    aligned: list[LabelSpan] = []

    for token in tokens:
        while (
            span_index < len(sorted_spans)
            and sorted_spans[span_index].end <= token.start
        ):
            span_index += 1

        label = "O"
        if span_index < len(sorted_spans):
            span = sorted_spans[span_index]
            if token.start < span.end and span.start < token.end:
                prefix = "I" if active_span_index == span_index else "B"
                active_span_index = span_index
                label = f"{prefix}-{span.label}"
            else:
                active_span_index = -1
        else:
            active_span_index = -1

        aligned.append(
            LabelSpan(
                start=token.start,
                end=token.end,
                text=token.text,
                label=label,
            )
        )

    for i in range(1, len(aligned) - 1):
        current = aligned[i]
        if current.label != "O":
            continue
        if current.text.strip() != "" and not re.fullmatch(
            r"[-/.,:;()\[\]{}<>]+", current.text
        ):
            continue

        left = aligned[i - 1]
        right = aligned[i + 1]
        if left.label.startswith(("B-", "I-")) and right.label.startswith(("B-", "I-")):
            left_label = left.label.split("-", 1)[1]
            right_label = right.label.split("-", 1)[1]
            if (
                left_label == right_label
                and current.start - left.end <= 1
                and right.start - current.end <= 1
            ):
                aligned[i] = LabelSpan(
                    start=current.start,
                    end=current.end,
                    text=current.text,
                    label=f"I-{left_label}",
                )

    return aligned


_EXCLUDED_BASE_RULES: set[tuple[str, int]] = {
    (ISO_LABEL_SSN, 51),
}

_MAX_BASE_RULES_PER_LABEL: dict[str, int] = {
    ISO_LABEL_EMAIL_ADDRESS: 1,
    ISO_LABEL_IP_ADDRESS: 1,
    ISO_LABEL_PCI: 5,
    ISO_LABEL_BANK_ACCOUNT: 4,
    ISO_LABEL_PHONE_NUMBER: 3,
    ISO_LABEL_SSN: 2,
    ISO_LABEL_PASSPORT: 5,
    ISO_LABEL_DRIVERS_LICENSE: 3,
    ISO_LABEL_VEHICLE_ID: 5,
}


def _build_lean_regex_rules() -> tuple[RegexRule, ...]:
    counts: defaultdict[str, int] = defaultdict(int)
    selected: list[RegexRule] = []

    for rule in REGEX_RULES:
        if rule.label not in NON_ML_FEASIBLE_LABELS:
            continue
        if (rule.label, rule.priority) in _EXCLUDED_BASE_RULES:
            continue

        max_rules = _MAX_BASE_RULES_PER_LABEL.get(rule.label, 2)
        if counts[rule.label] >= max_rules:
            continue

        selected.append(rule)
        counts[rule.label] += 1

    return tuple(selected)


LEAN_REGEX_RULES: tuple[RegexRule, ...] = _build_lean_regex_rules()


VIBE_ALPHA_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        label=ISO_LABEL_URL,
        pattern=_compile(r"(?P<value>\b(?:https?://|www\.)[^\s<>'\"\\)\]\}]+)"),
        validator=_is_probable_url,
        priority=120,
    ),
    RegexRule(
        label=ISO_LABEL_EMAIL_ADDRESS,
        pattern=_compile(
            r"""
            (?P<value>
                \b[\w.!#$%&'*+/=?^`{|}~-]+@[\w.-]+\.[A-Za-z]{2,24}\b
            )
            """
        ),
        validator=_is_probable_email,
        priority=11,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:
                credit\s+card(?:\s+number)?
                |debit\s+card(?:\s+number)?
                |card(?:\s+number)?
                |kreditkort(?:nummer(?:et)?)?
                |kortnummer(?:et)?
                |betalingskort(?:nummer(?:et)?)?
            )\b
            [^\n\r\d]{0,40}?
            (?:(?:ender\s+med|slutter\s+med|last\s*(?:digits?|4)|ending\s+with)\s*)?
            (?P<value>\d{3,4})\b
            """
        ),
        priority=33,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:cvv|cvc|security\s+code|kortets?\s+sikkerhedskode)\b
            [^\n\r\d]{0,20}?
            (?P<value>\d{3,4})\b
            """
        ),
        priority=34,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:
                kreditkort(?:oplysninger|nummer)?
                |credit\s+card(?:\s+number)?
                |betalings(?:oplysninger|information|metode)?
                |kortnummer
            )\b
            [^\n\r]{0,45}?
            (?P<value>\d{12,19})\b
            """
        ),
        priority=34,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:exp(?:iry|iration)?|expires?|udl\u00f8b(?:sdato)?)\b
            [^\n\r\d]{0,20}?
            (?P<value>(?:0?[1-9]|1[0-2])[/-](?:\d{2}|\d{4}))\b
            """
        ),
        priority=35,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:\d[ -]*?){13,19}\b
            [^\n\r]{0,10}?
            (?P<value>(?:0?[1-9]|1[0-2])[/-](?:\d{2}|\d{4}))\b
            """
        ),
        priority=35,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:\d[ -]*?){13,19}\b
            [^\n\r]{0,20}?
            \b(?:cvv|cvc)?\b
            [^\n\r\d]{0,8}?
            (?P<value>\d{3,4})\b
            """
        ),
        priority=36,
    ),
    RegexRule(
        label=ISO_LABEL_PHONE_NUMBER,
        pattern=_compile(
            r"""
            (?P<value>
                \+\d{1,3}(?:[\s().-]*\d){6,14}
            )
            """
        ),
        priority=58,
    ),
    RegexRule(
        label=ISO_LABEL_PHONE_NUMBER,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:\+?\d{2,4}(?:[.-]\d{2,5}){2,4})\b
            )
            """
        ),
        validator=_looks_like_phone_number_value,
        priority=59,
    ),
    RegexRule(
        label=ISO_LABEL_PHONE_NUMBER,
        pattern=_compile(
            r"""
            \b(?:tlf|telefon|mobil|phone|telefonnummer|mobilnummer)\b
            [^\n\r\d]{0,24}?
            (?P<value>\+?\d[\d .-]{6,22}\d)\b
            """
        ),
        validator=_looks_like_phone_number_value,
        priority=57,
    ),
    RegexRule(
        label=ISO_LABEL_SSN,
        pattern=_compile(
            r"""
            \b(?:
                cpr(?:-?nummer|nr\.?)?
                |personnummer
                |social(?:sikrings)?nummer
                |social\s*security\s*number
                |national(?:t)?[\-\s]?id
                |folkeregistreringsnummer
            )\b
            [^\n\r\d]{0,40}?
            (?P<value>\d{6}[- ]?\d{4}[A-Z]?)
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_looks_like_ssn,
        priority=51,
    ),
    RegexRule(
        label=ISO_LABEL_PASSPORT,
        pattern=_compile(
            r"""
            \b(?:
                passport(?:\s*(?:number|no\.?|id))?
                |pass(?:port)?\s*no\.?
                |passportnummer
                |passnummer
                |passnr\.?
                |pasnummer(?:et)?
                |pasnr\.?
                |rejsepas
            )\b
            [^\n\r]{0,50}?
            (?P<value>
                (?:[A-Z]{0,2}\d{6,10}[A-Z]?)
                |
                (?:\d{8,10})
                |
                (?:[A-Z][A-Z0-9]{6,10})
            )
            """
        ),
        validator=_looks_like_passport,
        priority=47,
    ),
    RegexRule(
        label=ISO_LABEL_PASSPORT,
        pattern=_compile(
            r"""
            (?P<value>
                \b[A-Z0-9]{8,10}<\d[A-Z]{2,3}\d{6,7}[A-Z]\d{6,7}\d<{8,}\d{2}\b
            )
            """
        ),
        validator=_looks_like_passport,
        priority=48,
    ),
    RegexRule(
        label=ISO_LABEL_VEHICLE_ID,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:[A-Z]{1,3}\s+\d{2}\s+\d{3}|[A-Z]{2}\s+\d{3}\s+\d{2})\b
            )
            """
        ),
        priority=94,
    ),
    RegexRule(
        label=ISO_LABEL_VEHICLE_ID,
        pattern=_compile(r"(?P<value>\b[A-Z]{2,3}\d{4,6}\b)"),
        validator=_looks_like_vehicle_id,
        priority=93,
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
                )/(?:\d{2}|\d{4})
                \b
            )
            """
        ),
        priority=133,
    ),
    RegexRule(
        label=ISO_LABEL_DATE,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:0?[1-9]|[12]\d|3[01])\.?\s+
                (?:
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
        priority=134,
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
                )\s+\d{1,2},\s*(?:19\d{2}|20\d{2})
                \b
            )
            """
        ),
        priority=135,
    ),
    RegexRule(
        label=ISO_LABEL_DATE,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:19\d{2}|20\d{2})-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])(?:t\d{2}:\d{2}:\d{2})?\b
            )
            """
        ),
        priority=136,
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
                    |l\u00f8rdag
                    |s\u00f8ndag
                    |lordag
                    |sondag
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
        priority=137,
    ),
    RegexRule(
        label=ISO_LABEL_DRIVERS_LICENSE,
        pattern=_compile(
            r"""
            \b(?:driver'?s?\s+licen[sc]e|driving\s+licen[sc]e|licen[sc]e\s+number|k\u00f8rekort(?:nummer)?|korekort(?:nummer)?|f\u00f8rerkort(?:\s*nr\.?)?)\b
            [^\n\r]{0,35}?
            (?P<value>(?:[A-Z]{1,4}[- ]?)?[A-Z0-9](?:[A-Z0-9-]{4,20}[A-Z0-9])?)
            """
        ),
        validator=_looks_like_drivers_license,
        priority=39,
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
                |skatte?(?:[\-\s]?(?:id|nummer|identifikationsnummer|personnummer|kortnummer|oplysninger))?
            )\b
            [^\n\r]{0,35}?
            (?P<value>
                (?:[A-Z]{2}[- ]?)?\d{6}[- ]?\d{4}
                |
                (?:[A-Z]{2}[- ]?)?\d{8,12}(?:[- ]?(?:DK|DS))?
                |
                \d{3,4}(?:[- ]\d{3,4}){2,3}
            )
            """
        ),
        validator=_is_valid_tax_number,
        priority=38,
    ),
    RegexRule(
        label=ISO_LABEL_TAX_NUMBER,
        pattern=_compile(
            r"""
            \b(?:tax(?:\s*(?:id|number))?|skatte?(?:id|nummer)|cvr(?:[\-\s]?nr\.?)?|moms(?:nummer)?)\b
            [^\n\r]{0,45}?
            (?P<value>(?:[A-Z]{2}[- ]?)?\d{8,12}(?:[- ]?(?:DK|DS))?|\d{6}[- ]\d{4})
            """
        ),
        validator=_is_valid_tax_number,
        priority=40,
    ),
    RegexRule(
        label=ISO_LABEL_TAX_NUMBER,
        pattern=_compile(
            r"(?P<value>\b(?:DK|DS)[- ]?\d{8,12}\b|\b\d{8,12}[- ]?(?:DK|DS)\b)"
        ),
        validator=_is_valid_tax_number,
        priority=42,
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
                (?:[A-Z]{2}[- ]?)?[A-Z0-9](?:[A-Z0-9-]{5,22}[A-Z0-9])?
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
            \b(?:id|id\s*card|id\s*kort)\b\s*[:#=\-\u2010-\u2015]\s*(?P<value>(?:DK[- ]?)?[A-Z0-9](?:[A-Z0-9-]{5,18}[A-Z0-9])?)
            """
        ),
        validator=_is_valid_id_card_number,
        priority=44,
    ),
    RegexRule(
        label=ISO_LABEL_ID_CARD_NUMBER,
        pattern=_compile(r"(?P<value>\b(?:DK|DS)[- ]?\d{8,12}\b)"),
        validator=_is_valid_id_card_number,
        priority=46,
    ),
    RegexRule(
        label=ISO_LABEL_ID_CARD_NUMBER,
        pattern=_compile(
            r"""
            \b(?:
                id(?:[\-\s]?kort(?:nummer)?)?
                |id\s*card(?:\s*number)?
                |identity\s*card
                |identitetskort(?:nummer)?
                |id\-nummer(?:et)?
            )\b
            [^\n\r]{0,35}?
            (?P<value>(?:[A-Z]{1,2}[- ]?)?\d(?:[ -]?\d){7,11}[A-Z]?)\b
            """
        ),
        validator=_is_valid_id_card_number,
        priority=45,
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
        for rule in (*LEAN_REGEX_RULES, *VIBE_ALPHA_RULES):
            if rule.label not in active_labels:
                continue
            for span in _iter_rule_spans(entry.text, rule):
                prioritized_spans.append((rule.priority, span))

        predicted_spans = _select_non_overlapping_spans(prioritized_spans)
        predicted_spans = _apply_context_relabeling(entry.text, predicted_spans)

        predicted_bio = _align_to_entry_tokens(entry.bio, predicted_spans)
        return PiiEntry(
            uid=entry.uid,
            text=entry.text,
            bio=predicted_bio,
            meta=entry.meta,
        )
