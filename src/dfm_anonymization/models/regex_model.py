import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

from ..constants import *
from ..dataset import PiiDataEntry
from ..interfaces import PiiModel
from ..util.util import Span, filter_bio, spans_to_bio

REGEX_COMPATIBLE_LABELS = {
    ISO_LABEL_BANK_ACCOUNT,
    ISO_LABEL_DRIVERS_LICENSE,
    ISO_LABEL_EMAIL_ADDRESS,
    ISO_LABEL_PASSPORT,
    ISO_LABEL_PHONE_NUMBER,
    ISO_LABEL_PCI,
    ISO_LABEL_IP_ADDRESS,
    ISO_LABEL_VEHICLE_ID,
    ISO_LABEL_SSN,
    ISO_LABEL_PRECISE_LOCATION,
}


RegexValidator = Callable[[str], bool]


@dataclass(frozen=True)
class RegexRule:
    label: str
    pattern: re.Pattern[str]
    group: str | int | None = "value"
    validator: RegexValidator | None = None
    priority: int = 100


def _compile(pattern: str, *, ignore_case: bool = True) -> re.Pattern[str]:
    flags = re.UNICODE | re.VERBOSE
    if ignore_case:
        flags |= re.IGNORECASE
    return re.compile(pattern, flags)


def _compact(value: str) -> str:
    return re.sub(r"[\s-]+", "", value)


def _digits(value: str) -> str:
    return re.sub(r"\D+", "", value)


def _is_luhn_valid(number: str) -> bool:
    digits = [int(ch) for ch in number]
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip())
        return True
    except ValueError:
        return False


def _is_valid_iban(value: str) -> bool:
    compact = _compact(value).upper()
    return (
        15 <= len(compact) <= 34
        and compact[:2].isalpha()
        and compact[2:4].isdigit()
        and compact.isalnum()
    )


def _is_valid_account_number(value: str) -> bool:
    compact = _compact(value).upper()
    digit_count = len(_digits(compact))
    return _is_valid_iban(value) or (compact.isalnum() and 4 <= digit_count <= 18)


def _is_valid_phone(value: str) -> bool:
    digits = _digits(value)
    if not 8 <= len(digits) <= 15:
        return False
    if 13 <= len(digits) <= 19 and _is_luhn_valid(digits):
        return False
    return any(sep in value for sep in "+-() ") or len(digits) == 8


def _is_valid_cpr(value: str) -> bool:
    compact = _digits(value)
    if len(compact) != 10:
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


def _is_valid_us_ssn(value: str) -> bool:
    compact = _digits(value)
    if len(compact) != 9:
        return False
    area = compact[:3]
    group = compact[3:5]
    serial = compact[5:]
    return area not in {"000", "666"} and not area.startswith("9") and group != "00" and serial != "0000"


def _is_valid_passport(value: str) -> bool:
    compact = _compact(value).upper()
    digit_count = len(_digits(compact))
    letter_count = sum(ch.isalpha() for ch in compact)
    return (
        compact.isalnum()
        and 6 <= len(compact) <= 12
        and 5 <= digit_count <= 10
        and letter_count >= 1
        and letter_count <= 4
    )


def _is_valid_drivers_license(value: str) -> bool:
    compact = _compact(value).upper()
    digit_count = len(_digits(compact))
    return compact.isalnum() and 5 <= len(compact) <= 20 and 5 <= digit_count <= 14


def _is_valid_vin(value: str) -> bool:
    compact = _compact(value).upper()
    for prefix in ("VIN", "CHASSISNR", "CHASSIS", "STELNUMMER"):
        if compact.startswith(prefix):
            compact = compact[len(prefix):]
            break
    return (
        len(compact) == 17
        and any(ch.isalpha() for ch in compact)
        and any(ch.isdigit() for ch in compact)
        and bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", compact))
    )


def _is_valid_plate_or_vin(value: str) -> bool:
    compact = _compact(value).upper()
    for prefix in ("VIN", "CHASSISNR", "CHASSIS", "STELNUMMER"):
        if compact.startswith(prefix):
            compact = compact[len(prefix):]
            break
    digit_count = len(_digits(compact))
    return (
        _is_valid_vin(compact)
        or bool(re.fullmatch(r"[A-Z]{2,3}\d{4,6}", compact))
        or bool(re.fullmatch(r"[A-Z]{1,6}\d{4,12}", compact))
        or (
            compact.isalnum()
            and 6 <= len(compact) <= 18
            and digit_count >= 2
            and any(ch.isalpha() for ch in compact)
        )
    )


REGEX_RULES: tuple[RegexRule, ...] = (
    RegexRule(
        label=ISO_LABEL_EMAIL_ADDRESS,
        pattern=_compile(
            r"""
            (?P<value>
                \b
                [a-z0-9.!#$%&'*+/=?^_`{|}~-]+
                @
                (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
                [a-z]{2,63}
                \b
            )
            """
        ),
        priority=10,
    ),
    RegexRule(
        label=ISO_LABEL_IP_ADDRESS,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b
                |
                (?<![:\w])[a-f0-9:]*:[a-f0-9:]+(?![:\w])
            )
            """
        ),
        validator=_is_valid_ip,
        priority=20,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(r"(?P<value>\b(?:\d[ -]*?){13,19}\b)"),
        validator=lambda value: 13 <= len(_digits(value)) <= 19 and _is_luhn_valid(_digits(value)),
        priority=30,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:
                card(?:\s+number)?
                |credit\s+card(?:\s+number)?
                |debit\s+card(?:\s+number)?
                |visa(?:-kort)?(?:\s*|-)?(?:nummer|number)?
                |mastercard(?:\s*|-)?(?:nummer|number)?
                |kreditkort(?:nummer(?:et)?)?
                |kortnummer(?:et)?
                |kortbetaling
                |kortoplysninger
                |kortnumre
                |kort(?:\s+med\s+nummer(?:et)?)?
            )\b
            \s*[:#-]?\s*
            (?P<value>(?:\d[ -]*?){13,19})
            """
        ),
        validator=lambda value: 13 <= len(_digits(value)) <= 19,
        priority=32,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:
                card|credit\s+card|debit\s+card|visa(?:-kort)?|mastercard|kreditkort|kort(?:nummer|betaling|oplysninger|numre)?
            )\b
            [^\n\r\d]{0,40}
            (?P<value>(?:\d[ -]*?){13,19})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=lambda value: 13 <= len(_digits(value)) <= 19,
        priority=34,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:cvv|cvc)\b\s*[:#-]?\s*(?P<value>\d{3,4})\b
            """
        ),
        priority=35,
    ),
    RegexRule(
        label=ISO_LABEL_PCI,
        pattern=_compile(
            r"""
            \b(?:udløbsdato|expiry|expiration\s+date|expires?)\b\s*[:#-]?\s*(?P<value>(?:0[1-9]|1[0-2])/[0-9]{2,4})\b
            """
        ),
        priority=36,
    ),
    RegexRule(
        label=ISO_LABEL_BANK_ACCOUNT,
        pattern=_compile(
            r"(?P<value>\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){2,7}[A-Z0-9]{0,2}\b)"
        ),
        validator=_is_valid_iban,
        priority=40,
    ),
    RegexRule(
        label=ISO_LABEL_BANK_ACCOUNT,
        pattern=_compile(
            r"""
            \b
            (?:account(?:\s+number)?|bank\s+account|acct\.?|account\s*no\.?|konto(?:nummer)?|kontonr\.?)
            \b\s*[:#-]?\s*
            (?P<value>[A-Z0-9][A-Z0-9\-/ ]{5,30}\d)
            """
        ),
        validator=_is_valid_account_number,
        priority=45,
    ),
    RegexRule(
        label=ISO_LABEL_BANK_ACCOUNT,
        pattern=_compile(
            r"""
            \b(?:bankkonto|kontonummer|kontooplysninger|konto\s+til\s+(?:overførsel|udbetaling|lønudbetaling|refusion)|bankforbindelse)\b
            [^\n\r]{0,30}?
            (?P<value>\d{4}[ -]?\d{7,10}|\d{7,10})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_account_number,
        priority=46,
    ),
    RegexRule(
        label=ISO_LABEL_BANK_ACCOUNT,
        pattern=_compile(
            r"""
            \b(?:reg(?:istrerings)?nr\.?\s*/?\s*konto(?:nr\.)?|reg\.nr\.?\s*/\s*kontonr\.?|bank\s*\(reg\.nr\./konto\))\b
            \s*[:#-]?\s*
            (?P<value>\d{4}[ -]?\d{7,10})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_account_number,
        priority=47,
    ),
    RegexRule(
        label=ISO_LABEL_BANK_ACCOUNT,
        pattern=_compile(
            r"""
            \breg(?:istrerings)?nr\.?\s*[:#-]?\s*(?P<value>\d{4})\b
            """
        ),
        validator=_is_valid_account_number,
        priority=47,
    ),
    RegexRule(
        label=ISO_LABEL_BANK_ACCOUNT,
        pattern=_compile(
            r"""
            \bkontonr\.?\s*[:#-]?\s*(?P<value>\d{7,10}|\d{4}[ -]\d{7,10})\b
            """
        ),
        validator=_is_valid_account_number,
        priority=47,
    ),
    RegexRule(
        label=ISO_LABEL_SSN,
        pattern=_compile(r"(?P<value>\b(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{2}[- ]?\d{4}\b)"),
        validator=_is_valid_cpr,
        priority=50,
    ),
    RegexRule(
        label=ISO_LABEL_SSN,
        pattern=_compile(
            r"(?P<value>\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b)"
        ),
        validator=_is_valid_us_ssn,
        priority=55,
    ),
    RegexRule(
        label=ISO_LABEL_PHONE_NUMBER,
        pattern=_compile(r"(?P<value>\b\d{8}\b)"),
        validator=_is_valid_phone,
        priority=60,
    ),
    RegexRule(
        label=ISO_LABEL_PHONE_NUMBER,
        pattern=_compile(
            r"(?P<value>(?<!\w)(?:\+\d{1,3}[ .-]*)?(?:\(?\d{2,4}\)?[ .-]*){2,4}\d{2,4}(?!\w))"
        ),
        validator=_is_valid_phone,
        priority=65,
    ),
    RegexRule(
        label=ISO_LABEL_PASSPORT,
        pattern=_compile(
            r"""
            \b(?:passport(?:\s+number|\s+no\.?)|passnummer|passnr\.?|pasnummer(?:et)?|pasnr\.?)\b
            \s*[:#-]?\s*
            (?P<value>[A-Z]{0,2}\d{6,9})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=lambda value: _compact(value).isalnum() and 6 <= len(_digits(value)) <= 9,
        priority=47,
    ),
    RegexRule(
        label=ISO_LABEL_PASSPORT,
        pattern=_compile(
            r"""
            \b(?:passport(?:\s+(?:number|no\.?|id|backup|copy|ved\s+check-in))?|passportnummer|passportkopi|passnummer|passnr\.?|pas(?:nummer(?:et)?|nr\.?|oplysninger?|information)?|rejsepas)\b
            \s*[:#-]?\s*
            (?P<value>[A-Z]{0,3}\d{6,9})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_passport,
        priority=48,
    ),
    RegexRule(
        label=ISO_LABEL_PASSPORT,
        pattern=_compile(
            r"""
            \b(?:passport|passportnummer|passportkopi|passnummer|passnr\.?|pas(?:nummer(?:et)?|nr\.?|oplysninger?|information)?|rejsepas)\b
            [^\n\r]{0,50}?
            (?P<value>[A-Z]{0,3}\d{6,9})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_passport,
        priority=49,
    ),
    RegexRule(
        label=ISO_LABEL_DRIVERS_LICENSE,
        pattern=_compile(
            r"""
            \b(?:driver'?s?\s+licen[sc]e|driving\s+licen[sc]e|licen[sc]e\s+number|driver\s+license(?:\s+id)?|førerkort(?:\s+nr\.?)?|kørekort(?:nummer)?|kørekort\s+nr\.?)\b
            \s*[:#-]?\s*
            (?P<value>[A-Z0-9]+(?:[- ][A-Z0-9]+)*)
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_drivers_license,
        priority=80,
    ),
    RegexRule(
        label=ISO_LABEL_DRIVERS_LICENSE,
        pattern=_compile(
            r"""
            \b(?:driver'?s?\s+licen[sc]e|driving\s+licen[sc]e|licen[sc]e\s+number|driver\s+license(?:\s+id)?|førerkort(?:\s+nr\.?)?|kørekort(?:nummer)?|kørekort\s+nr\.?)\w*\b
            (?:\s+(?:er|var|som|blev|oplyst(?:e)?|registreret(?:\s+som)?|fremvist|fundet|nummer(?:et)?))?
            \s*[:#-]?\s*
            (?P<value>(?:[A-Z]{1,6}[- ]?)?\d{5,12}|[A-Z]{1,6}(?:[- ]?[A-Z0-9]{2,}){2,})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_drivers_license,
        priority=82,
    ),
    RegexRule(
        label=ISO_LABEL_VEHICLE_ID,
        pattern=_compile(r"(?P<value>\b[A-HJ-NPR-Z0-9]{17}\b)"),
        validator=_is_valid_vin,
        priority=90,
    ),
    RegexRule(
        label=ISO_LABEL_VEHICLE_ID,
        pattern=_compile(
            r"""
            \b(?:license\s+plate|number\s+plate|registration(?:\s+number)?|vehicle\s+id|vin|nummerplade|registreringsnummer)\b
            \s*[:#-]?\s*
            (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:[A-Z]{1,6}[ -]?\d{4,12})|(?:[A-Z0-9]{8,20}))
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_plate_or_vin,
        priority=95,
    ),
    RegexRule(
        label=ISO_LABEL_VEHICLE_ID,
        pattern=_compile(
            r"""
            \b(?:køretøj(?:et)?|firmabil(?:en|ens)?|udlejningsbil(?:en|ens)?|nummerplade(?:n)?|reg(?:istrerings)?nr\.?|registreringsnummer(?:\s+for\s+køretøjet)?|leasede\s+køretøjer|trailer\s+id|stelnummer|chassisnr\.?)\b
            (?:\s+(?:er|var|som|med|på|for|:|nr\.?|nummer(?:plade)?|registreret(?:\s+som)?))?
            \s*[:#-]?\s*
            (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:[A-Z]{1,8}[ -]?\d{4,12})|(?:[A-Z0-9]{8,20}))
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_plate_or_vin,
        priority=96,
    ),
    RegexRule(
        label=ISO_LABEL_VEHICLE_ID,
        pattern=_compile(
            r"""
            \b(?:bilen\s+.*?registreringsnummer|bilens\s+registreringsnummer|bilnummerplade|firmabilernes\s+nummerplader|fragtbil|lastbilplade|nummerplade(?:n)?|registreringsnummer(?:\s*\(nummerplade\))?|stelnummer|chassisnr\.?)\b
            [^\n\r]{0,45}?
            (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:VIN)?[A-Z0-9]{8,20})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_plate_or_vin,
        priority=97,
    ),
    RegexRule(
        label=ISO_LABEL_VEHICLE_ID,
        pattern=_compile(
            r"""
            \b(?:bilen|bilens|firmabil(?:en|ens)?|fragtbil(?:en)?|lastbilen)\b
            [^\n\r]{0,50}?\b(?:registreringsnummer|nummerplade|plade|registreret\s+som)\b
            [^\n\r]{0,20}?
            (?P<value>(?:[A-Z]{2,3}(?:[ -]?\d){4,6})|(?:VIN)?[A-Z0-9]{8,20})
            (?=$|[^A-Za-z0-9])
            """
        ),
        validator=_is_valid_plate_or_vin,
        priority=98,
    ),
    RegexRule(
        label=ISO_LABEL_PRECISE_LOCATION,
        pattern=_compile(
            r"""
            (?P<value>
                \b[+-]?(?:90(?:\.0+)?|[1-8]?\d\.\d{3,})\s*[,;/]\s*[+-]?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d{1,2}\.\d{3,})\b
            )
            """
        ),
        priority=100,
    ),
    RegexRule(
        label=ISO_LABEL_PRECISE_LOCATION,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:[0-8]?\d(?:\.\d+)?|90(?:\.0+)?)°?\s*[NS]\s*[,;/]\s*(?:[0-9]{1,3}(?:\.\d+)?|1[0-7]\d(?:\.\d+)?|180(?:\.0+)?)°?\s*[EW]\b
            )
            """
        ),
        priority=101,
    ),
    RegexRule(
        label=ISO_LABEL_PRECISE_LOCATION,
        pattern=_compile(
            r"""
            (?P<value>
                \b\d{1,5}[A-Z]?\s+
                (?:[A-ZÆØÅ][\w'.-]+\s+){0,4}
                (?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|way|court|ct\.?|place|pl\.?|terrace|ter\.?|parkway|pkwy\.?|highway|hwy\.?)
                \b(?:,?\s+[A-ZÆØÅ][\w'.-]+){0,3}(?:,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?)?
            )
            """,
            ignore_case=False,
        ),
        priority=105,
    ),
    RegexRule(
        label=ISO_LABEL_PRECISE_LOCATION,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:[A-ZÆØÅ][A-Za-zÆØÅæøå'.-]*\s+){1,5}
                \d{1,5}[A-Z]?
                (?:,\s*(?:st\.?|th\.?|tv\.?|mf\.?|\d+\.\s*(?:th\.?|tv\.?|mf\.?)))?
                (?:,\s*|\s+)\d{4}\s+[A-ZÆØÅ][A-Za-zÆØÅæøå.-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå.-]*|[A-ZÆØÅ])){0,3}
            )
            """,
            ignore_case=False,
        ),
        priority=108,
    ),
    RegexRule(
        label=ISO_LABEL_PRECISE_LOCATION,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:[A-ZÆØÅ][\w'.-]+\s+){0,4}
                (?:
                    [A-ZÆØÅ][\w'.-]*(?:gade|vej|vejen|alle|allé|boulevard|bakken|draget|stræde|stien|sti|havn|bro)
                    |
                    (?:[A-ZÆØÅ][\w'.-]+\s+)?(?:Plads|Torv|Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Way|Court|Ct\.?|Place|Pl\.?|Terrace|Ter\.?|Parkway|Pkwy\.?|Highway|Hwy\.?)
                )
                \s+\d{1,5}[A-Z]?
                (?:,\s*\d{4}\s*[A-ZÆØÅ][A-Za-zÆØÅæøå-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå-]*|[A-ZÆØÅ])){0,3})?
            )
            """,
            ignore_case=False,
        ),
        priority=110,
    ),
    RegexRule(
        label=ISO_LABEL_PRECISE_LOCATION,
        pattern=_compile(
            r"""
            (?P<value>
                \bpostboks\s+\d{1,5}(?:,\s*|\s+)\d{4}\s+[A-ZÆØÅ][A-Za-zÆØÅæøå.-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå.-]*|[A-ZÆØÅ])){0,3}
            )
            """
        ),
        priority=112,
    ),
    RegexRule(
        label=ISO_LABEL_PRECISE_LOCATION,
        pattern=_compile(
            r"""
            (?P<value>
                \b(?:[A-ZÆØÅ][A-Za-zÆØÅæøå'.-]*(?:gade|vej|vejen|allé|alle|boulevard|plads|torv|stræde|sti|havn|bro)|[A-ZÆØÅ][A-Za-zÆØÅæøå'.-]+(?:\s+[A-ZÆØÅ][A-Za-zÆØÅæøå'.-]+){0,2})
                \s+og\s+
                (?:[A-ZÆØÅ][A-Za-zÆØÅæøå'.-]*(?:gade|vej|vejen|allé|alle|boulevard|plads|torv|stræde|sti|havn|bro)|[A-ZÆØÅ][A-Za-zÆØÅæøå'.-]+(?:\s+[A-ZÆØÅ][A-Za-zÆØÅæøå'.-]+){0,2})
                ,\s*\d{4}\s+[A-ZÆØÅ][A-Za-zÆØÅæøå.-]*(?:\s+(?:[A-ZÆØÅ][A-Za-zÆØÅæøå.-]*|[A-ZÆØÅ])){0,3}
            )
            """,
            ignore_case=False,
        ),
        priority=114,
    ),
)


def _iter_rule_spans(text: str, rule: RegexRule) -> Iterable[Span]:
    for match in rule.pattern.finditer(text):
        start, end = match.span(rule.group) if rule.group is not None else match.span()
        candidate = text[start:end].strip(" ,;:\n\t")
        if not candidate:
            continue
        if rule.validator is not None and not rule.validator(candidate):
            continue
        candidate_start = text.find(candidate, start, end)
        yield Span(
            start=candidate_start if candidate_start >= 0 else start,
            end=(candidate_start + len(candidate)) if candidate_start >= 0 else end,
            label=rule.label,
        )


def _select_non_overlapping_spans(spans: list[tuple[int, Span]]) -> list[Span]:
    selected: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for _, span in sorted(spans, key=lambda item: (item[1].start, item[0], -(item[1].end - item[1].start))):
        key = (span.start, span.end, span.label)
        if key in seen:
            continue
        if any(span.start < existing.end and existing.start < span.end for existing in selected):
            continue
        selected.append(span)
        seen.add(key)
    return selected


class RegexModel(PiiModel):
    def __init__(
        self,
        name: str = "regex",
    ):
        super().__init__(name)

    def __call__(
        self, entry: PiiDataEntry, label_whitelist: set[str] | None = None
    ) -> list[str]:
        active_labels = REGEX_COMPATIBLE_LABELS
        if label_whitelist is not None:
            active_labels = REGEX_COMPATIBLE_LABELS & label_whitelist

        prioritized_spans: list[tuple[int, Span]] = []
        for rule in REGEX_RULES:
            if rule.label not in active_labels:
                continue
            for span in _iter_rule_spans(entry.text, rule):
                prioritized_spans.append((rule.priority, span))

        token_spans = entry.get_token_spans()
        bio_tags = spans_to_bio(
            token_spans=token_spans,
            spans=_select_non_overlapping_spans(prioritized_spans),
        )
        return filter_bio(bio_tags, active_labels) if label_whitelist else bio_tags
