import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from pii_eval.constants import ISO_LABEL_PERSON_NAME
from pii_eval.data import PiiEntry, PiiModel
from pii_eval.span import LabelSpan, spans_to_bio


# This should be a model solely targeting the PERSON NAME label
# and solely based on looking up in lists found in
# src/pii_eval/assets/person_names/<first/last>_names_da.txt
# It should ensure be somewhat case independent and also catch
# cases like <first-name>     <last-name> or
# <last-name>, <first-name> ... etc. fill in more examples like this

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", flags=re.UNICODE)
FIELD_PREFIX_PATTERN = re.compile(
    r"(?:fornavn|efternavn|firstname|lastname|given\s*name|surname|full\s*name|name)\s*[:\-]\s*$",
    flags=re.IGNORECASE,
)

TITLE_OR_GREETINGS = {
    "hr",
    "fru",
    "frk",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "professor",
    "herr",
    "madam",
    "sir",
    "dear",
    "hej",
    "kære",
}

FIELD_MARKER_TOKENS = {
    "fornavn",
    "efternavn",
    "firstname",
    "lastname",
    "givenname",
    "surname",
    "fullname",
    "full",
    "name",
    "navn",
}


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _normalize_token(value: str) -> str:
    return _normalize_text(value).casefold()


def _tokenize_name(raw_name: str) -> tuple[str, ...]:
    normalized = _normalize_text(raw_name).strip()
    tokens = [
        _normalize_token(match.group(0)) for match in WORD_PATTERN.finditer(normalized)
    ]
    return tuple(tokens)


def _is_whitespace_gap(text: str) -> bool:
    return bool(text) and text.strip() == ""


def _is_comma_gap(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped == ","


class _TextToken:
    __slots__ = ("start", "end", "text", "norm")

    def __init__(self, start: int, end: int, text: str, norm: str):
        self.start = start
        self.end = end
        self.text = text
        self.norm = norm


class PNLookup(PiiModel):
    def __init__(
        self,
        name: str = "pn-lookup",
    ):
        super().__init__(name)
        self._first_by_len: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        self._last_by_len: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        self._first_lengths: list[int] = []
        self._last_lengths: list[int] = []
        self._first_single_tokens: set[str] = set()
        self._last_single_tokens: set[str] = set()
        self._lookup_single_tokens: set[str] = set()
        self._load_name_lookup()

    def _load_name_lookup(self) -> None:
        base_path = Path(__file__).resolve().parents[1] / "assets" / "person_names"
        first_names_path = base_path / "first_names_da.txt"
        last_names_path = base_path / "last_names_da.txt"

        self._first_by_len = self._load_names_by_len(first_names_path)
        self._last_by_len = self._load_names_by_len(last_names_path)
        self._first_lengths = sorted(self._first_by_len.keys(), reverse=True)
        self._last_lengths = sorted(self._last_by_len.keys(), reverse=True)
        self._first_single_tokens = {
            name[0] for name in self._first_by_len.get(1, set())
        }
        self._last_single_tokens = {name[0] for name in self._last_by_len.get(1, set())}
        self._lookup_single_tokens = (
            self._first_single_tokens | self._last_single_tokens
        )

    def _load_names_by_len(self, path: Path) -> dict[int, set[tuple[str, ...]]]:
        by_len: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        for line in path.read_text(encoding="utf-8").splitlines():
            name_tokens = _tokenize_name(line)
            if not name_tokens:
                continue

            if len(name_tokens) == 1 and name_tokens[0] in TITLE_OR_GREETINGS:
                continue

            # Ignore single-character names to avoid spurious matches.
            if len(name_tokens) == 1 and len(name_tokens[0]) < 2:
                continue

            by_len[len(name_tokens)].add(name_tokens)
        return by_len

    def _tokenize_text(self, text: str) -> list[_TextToken]:
        return [
            _TextToken(
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                norm=_normalize_token(match.group(0)),
            )
            for match in WORD_PATTERN.finditer(text)
        ]

    def _is_name_cased_token(self, token: _TextToken) -> bool:
        if len(token.text) < 2:
            return False

        for part in re.split(r"[-']", token.text):
            if not part:
                return False
            if not part[0].isupper():
                return False
            if not all(ch.isalpha() for ch in part):
                return False

        return True

    def _capitalized_run_end(
        self,
        text: str,
        tokens: list[_TextToken],
        start: int,
        max_tokens: int = 6,
    ) -> int:
        end = start
        token_count = len(tokens)

        while end < token_count and end - start < max_tokens:
            if tokens[end].norm in FIELD_MARKER_TOKENS:
                break

            if not self._is_name_cased_token(tokens[end]):
                break

            if end > start:
                gap = text[tokens[end - 1].end : tokens[end].start]
                if not _is_whitespace_gap(gap):
                    break

            end += 1

        return end

    def _centered_capitalized_run(
        self,
        text: str,
        tokens: list[_TextToken],
        center: int,
        max_tokens: int = 6,
    ) -> tuple[int, int]:
        left = center
        right = center + 1
        token_count = len(tokens)

        while right < token_count and right - left < max_tokens:
            token = tokens[right]
            if token.norm in FIELD_MARKER_TOKENS or token.norm in TITLE_OR_GREETINGS:
                break
            if not self._is_name_cased_token(token):
                break

            gap = text[tokens[right - 1].end : token.start]
            if not _is_whitespace_gap(gap):
                break

            right += 1

        while left > 0 and right - left < max_tokens:
            token = tokens[left - 1]
            if token.norm in FIELD_MARKER_TOKENS or token.norm in TITLE_OR_GREETINGS:
                break
            if not self._is_name_cased_token(token):
                break

            gap = text[token.end : tokens[left].start]
            if not _is_whitespace_gap(gap):
                break

            left -= 1

        return left, right

    def _has_field_prefix(self, text: str, start: int, window: int = 48) -> bool:
        prefix = text[max(0, start - window) : start]
        normalized = _normalize_text(prefix)
        return bool(FIELD_PREFIX_PATTERN.search(normalized))

    def _has_title_or_greeting_prefix(
        self, text: str, tokens: list[_TextToken], index: int
    ) -> bool:
        if index < 1:
            return False

        previous = tokens[index - 1]
        gap = text[previous.end : tokens[index].start]
        if not _is_whitespace_gap(gap):
            return False

        return previous.norm in TITLE_OR_GREETINGS

    def _gaps_are_whitespace(
        self, text: str, tokens: list[_TextToken], start: int, end: int
    ) -> bool:
        if end - start <= 1:
            return True

        for idx in range(start, end - 1):
            gap = text[tokens[idx].end : tokens[idx + 1].start]
            if not _is_whitespace_gap(gap):
                return False

        return True

    def _match_name_sequences(
        self,
        text: str,
        tokens: list[_TextToken],
        index: int,
        lengths: list[int],
        lookup: dict[int, set[tuple[str, ...]]],
    ) -> list[tuple[int, int]]:
        matches: list[tuple[int, int]] = []
        token_count = len(tokens)

        for length in lengths:
            end = index + length
            if end > token_count:
                continue

            candidate = tuple(token.norm for token in tokens[index:end])
            if candidate not in lookup.get(length, set()):
                continue

            if not self._gaps_are_whitespace(text, tokens, index, end):
                continue

            matches.append((index, end))

        return matches

    def __call__(self, entry: PiiEntry) -> PiiEntry:
        return PiiEntry(
            uid=entry.uid,
            text=entry.text,
            bio=self.predict_bio(entry.text, entry.bio),
            meta=entry.meta,
        )

    def predict_spans(self, text):
        return self.predict_spans(text)

    def predict_bio(self, text, _bio):
        return spans_to_bio(text, self.predict_spans(text))

    def predict_spans(self, text: str) -> list[tuple[int, int]]:
        tokens = self._tokenize_text(text)
        token_count = len(tokens)
        if token_count < 1:
            return []

        candidates: list[tuple[int, int]] = []

        for index in range(token_count):
            # Always match lookup names when cased like names, and allow
            # lowercase long surnames (e.g. "jacobsen") as a recall fallback.
            if (
                tokens[index].norm in self._lookup_single_tokens
                and tokens[index].norm not in TITLE_OR_GREETINGS
                and (
                    self._is_name_cased_token(tokens[index])
                    or (
                        tokens[index].norm in self._last_single_tokens
                        and len(tokens[index].norm) >= 5
                    )
                )
            ):
                candidates.append((tokens[index].start, tokens[index].end))

            first_matches = self._match_name_sequences(
                text=text,
                tokens=tokens,
                index=index,
                lengths=self._first_lengths,
                lookup=self._first_by_len,
            )
            for first_start, first_end in first_matches:
                if first_end >= token_count:
                    continue

                delimiter = text[tokens[first_end - 1].end : tokens[first_end].start]
                if not _is_whitespace_gap(delimiter):
                    continue

                last_matches = self._match_name_sequences(
                    text=text,
                    tokens=tokens,
                    index=first_end,
                    lengths=self._last_lengths,
                    lookup=self._last_by_len,
                )
                for _, last_end in last_matches:
                    candidates.append(
                        (tokens[first_start].start, tokens[last_end - 1].end)
                    )
                    break

            last_matches = self._match_name_sequences(
                text=text,
                tokens=tokens,
                index=index,
                lengths=self._last_lengths,
                lookup=self._last_by_len,
            )
            for last_start, last_end in last_matches:
                if last_end >= token_count:
                    continue

                delimiter = text[tokens[last_end - 1].end : tokens[last_end].start]
                if not _is_comma_gap(delimiter):
                    continue

                reversed_first_matches = self._match_name_sequences(
                    text=text,
                    tokens=tokens,
                    index=last_end,
                    lengths=self._first_lengths,
                    lookup=self._first_by_len,
                )
                for _, first_end in reversed_first_matches:
                    candidates.append(
                        (tokens[last_start].start, tokens[first_end - 1].end)
                    )
                    break

            # Lookup-anchored sequence expansion catches names with middle tokens.
            if (
                tokens[index].norm in self._first_single_tokens
                and tokens[index].norm not in TITLE_OR_GREETINGS
            ):
                run_end = self._capitalized_run_end(
                    text=text, tokens=tokens, start=index
                )
                if run_end - index >= 2:
                    candidates.append((tokens[index].start, tokens[run_end - 1].end))

            # Expand around any known lookup token to recover names where only
            # middle/last components are in the dictionaries.
            if (
                tokens[index].norm in self._lookup_single_tokens
                and tokens[index].norm not in TITLE_OR_GREETINGS
                and self._is_name_cased_token(tokens[index])
            ):
                run_start, run_end = self._centered_capitalized_run(
                    text=text,
                    tokens=tokens,
                    center=index,
                )
                if run_end - run_start >= 2:
                    candidates.append(
                        (tokens[run_start].start, tokens[run_end - 1].end)
                    )

            # Keep contextual checks for expansion logic above; single-token
            # lookup matches are handled unconditionally at loop start.

        # Prefer longer spans when multiple matches start at the same position.
        selected: list[tuple[int, int]] = []
        for start, end in sorted(
            set(candidates), key=lambda span: (span[0], -(span[1] - span[0]))
        ):
            if any(
                start < existing_end and existing_start < end
                for existing_start, existing_end in selected
            ):
                continue
            selected.append((start, end))

        return selected
