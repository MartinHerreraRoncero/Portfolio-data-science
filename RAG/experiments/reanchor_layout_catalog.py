"""Layout catalog reanchoring with exact raw offsets and provenance validation."""

import copy
import hashlib
import unicodedata

from experiments.preserve_evidence import collapse_spaces, validate_catalog

TYPOGRAPHIC_DASHES = {
    '\u2010',  # Hyphen
    '\u2011',  # Non-breaking hyphen
    '\u2012',  # Figure dash
    '\u2013',  # En dash
    '\u2014',  # Em dash
    '\u2015',  # Horizontal bar
    '\u2212',  # Minus sign
}

NORMALIZATION_RULE = (
    "Unicode NFD per character; remove whitespace; map finite typographic dashes "
    "U+2010..2015 and U+2212 to ASCII '-'; retain accents, case, digits, decimal comma, %, "
    "and other punctuation; unique full normalized match including overlapping occurrences"
)


def canonical_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Map text to normalized characters and their source raw [start, end) offsets.

    Applies Unicode NFD per character, strips whitespace, and maps finite
    typographic dashes (U+2010..2015, U+2212) to ASCII '-'. Retains accents,
    case, digits, decimal commas, percentages, and all other punctuation.
    """
    if type(text) is not str:
        raise ValueError(f"text must be str, got {type(text).__name__}")
    if not text:
        raise ValueError("text cannot be empty")
    if not text.strip():
        raise ValueError("text cannot be blank or only whitespace")

    norm_chars: list[str] = []
    offset_map: list[tuple[int, int]] = []

    for i, char in enumerate(text):
        if char.isspace():
            continue
        if char in TYPOGRAPHIC_DASHES:
            norm_chars.append('-')
            offset_map.append((i, i + 1))
        else:
            decomposed = unicodedata.normalize('NFD', char)
            for d in decomposed:
                if d.isspace():
                    continue
                norm_chars.append(d)
                offset_map.append((i, i + 1))

    return "".join(norm_chars), offset_map


def reanchor_span(
    quote: str,
    page_text: str,
    unit_id: str | None = None,
    page_number: int | None = None,
) -> tuple[int, int]:
    """Find unique raw character span [start, end) for quote in page_text.

    Normalized comparison ignores whitespace differences and unifies finite
    typographic dashes, but returned offsets index the exact raw page_text slice.
    Matches must align with canonical grapheme boundaries and recanonicalize
    identically to the quote key. Overlapping matches are checked to ensure uniqueness.
    """
    if type(quote) is not str:
        raise ValueError(f"quote must be str, got {type(quote).__name__}")
    if not quote or not quote.strip():
        raise ValueError("quote must be a nonempty string")
    if type(page_text) is not str:
        raise ValueError(f"page_text must be str, got {type(page_text).__name__}")
    if not page_text or not page_text.strip():
        raise ValueError("page_text must be a nonempty string")

    unit_context = f" for unit '{unit_id}'" if unit_id is not None else ""
    page_context = f" on page {page_number}" if page_number is not None else ""

    norm_quote, _ = canonical_map(quote)
    if not norm_quote:
        raise ValueError(f"Normalized quote is empty{unit_context}{page_context}")

    if unicodedata.combining(norm_quote[0]) != 0:
        raise ValueError(
            f"Quote starts with a standalone combining mark{unit_context}{page_context}: {quote!r}"
        )

    norm_page, offset_map = canonical_map(page_text)
    if not norm_page:
        raise ValueError(f"Normalized page_text is empty{unit_context}{page_context}")

    matches: list[tuple[int, int]] = []
    start = 0
    q_len = len(norm_quote)
    page_len = len(norm_page)

    while True:
        pos = norm_page.find(norm_quote, start)
        if pos == -1:
            break
        start = pos + 1

        # Complete canonical grapheme boundary checks:
        # Match cannot start in the middle of a grapheme cluster
        if pos > 0 and unicodedata.combining(norm_page[pos]) != 0:
            continue
        # Match cannot end in the middle of a grapheme cluster
        end_pos = pos + q_len
        if end_pos < page_len and unicodedata.combining(norm_page[end_pos]) != 0:
            continue

        match_start = pos
        match_end = end_pos - 1
        start_char = offset_map[match_start][0]
        end_char = offset_map[match_end][1]

        if (
            type(start_char) is not int
            or type(end_char) is not int
            or not (0 <= start_char < end_char <= len(page_text))
        ):
            continue

        # Recanonicalize projected raw slice to verify exact match with full quote key
        raw_slice = page_text[start_char:end_char]
        try:
            norm_raw, _ = canonical_map(raw_slice)
        except ValueError:
            continue

        if norm_raw != norm_quote:
            continue

        matches.append((start_char, end_char))

    if len(matches) == 0:
        raise ValueError(
            f"Normalized quote not found{unit_context}{page_context}: {quote!r}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous normalized quote found {len(matches)} times"
            f"{unit_context}{page_context}: {quote!r}"
        )

    return matches[0]


def _validate_pages_input(name: str, pages: list[dict]) -> None:
    if type(pages) is not list or not pages:
        raise ValueError(f"{name} must be a nonempty list")
    for p in pages:
        if type(p) is not dict:
            raise ValueError(f"Each item in {name} must be a dict")
        for fld in ("doc_id", "source_file", "published_at", "text"):
            val = p.get(fld)
            if type(val) is not str or not val:
                raise ValueError(f"Page field '{fld}' must be a nonempty string in {name}")
        pnum = p.get("page_number")
        if type(pnum) is not int:  # rejects bool
            raise ValueError(f"Page field 'page_number' must be int (not bool) in {name}")
        if pnum < 1:
            raise ValueError(f"Invalid page_number {pnum} in {name}")


def _validate_source_catalog_input(catalog: dict) -> None:
    if type(catalog) is not dict:
        raise ValueError("source_catalog must be a dict")
    units = catalog.get("units")
    if type(units) is not list or not units:
        raise ValueError("source_catalog units must be a nonempty list")
    seen_ids = set()
    for u in units:
        if type(u) is not dict:
            raise ValueError("Unit must be a dict")
        uid = u.get("id")
        if type(uid) is not str or not uid or uid in seen_ids:
            raise ValueError(f"Invalid or duplicate unit id: {uid!r}")
        seen_ids.add(uid)
        if type(u.get("kind")) is not str:
            raise ValueError(f"Unit '{uid}' kind must be str")
        if type(u.get("complete_end")) is not bool:
            raise ValueError(f"Unit '{uid}' complete_end must be bool")
        spans = u.get("spans")
        if type(spans) is not list or not spans:
            raise ValueError(f"Unit '{uid}' spans must be a nonempty list")
        for s in spans:
            if type(s) is not dict:
                raise ValueError(f"Span in unit '{uid}' must be a dict")
            for fld in ("doc_id", "source_file", "published_at", "quote"):
                val = s.get(fld)
                if type(val) is not str or not val:
                    raise ValueError(f"Span field '{fld}' must be a nonempty string in unit '{uid}'")
            for fld in ("start_char", "end_char", "page_number"):
                val = s.get(fld)
                if type(val) is not int:  # rejects bool
                    raise ValueError(f"Span field '{fld}' must be int (not bool) in unit '{uid}'")


def reanchor_catalog(
    source_catalog: dict,
    source_pages: list[dict],
    target_pages: list[dict],
) -> tuple[dict, dict]:
    """Reanchor source_catalog literal spans against target_pages.

    Validates source_catalog with validate_catalog before mapping and validate_catalog
    on the new catalog after mapping. Preserves all semantic units, spans, order, and
    metadata while updating start_char, end_char, and quote to exact target text.
    """
    _validate_source_catalog_input(source_catalog)
    _validate_pages_input("source_pages", source_pages)
    _validate_pages_input("target_pages", target_pages)

    # Check duplicate pages in source_pages and target_pages
    source_keys = set()
    for p in source_pages:
        k = (p["doc_id"], p["page_number"])
        if k in source_keys:
            raise ValueError(f"Duplicate page in source_pages: {k}")
        source_keys.add(k)

    target_keys = set()
    for p in target_pages:
        k = (p["doc_id"], p["page_number"])
        if k in target_keys:
            raise ValueError(f"Duplicate page in target_pages: {k}")
        target_keys.add(k)

    # Identical unique page identity sets required
    source_idents = {
        (p["doc_id"], p["source_file"], p["published_at"], p["page_number"])
        for p in source_pages
    }
    target_idents = {
        (p["doc_id"], p["source_file"], p["published_at"], p["page_number"])
        for p in target_pages
    }
    if source_idents != target_idents:
        missing_in_target = source_idents - target_idents
        extra_in_target = target_idents - source_idents
        raise ValueError(
            f"Identical unique page identity sets required. "
            f"Missing in target: {missing_in_target}, Extra in target: {extra_in_target}"
        )

    # Validate source catalog before mapping
    validate_catalog(source_catalog, source_pages)

    target_page_mapping = {(p["doc_id"], p["page_number"]): p for p in target_pages}

    # Deepcopy to preserve original input objects completely
    new_catalog = copy.deepcopy(source_catalog)
    new_catalog["profile"] = f"reanchored ({source_catalog.get('profile', '')})".strip()
    new_catalog["scope"] = "reanchored layout catalog; not a universal PDF parser"
    new_catalog["audit"] = "reanchored"
    new_catalog.pop("diagnostics", None)

    span_audits: list[dict] = []
    glyph_differences: list[dict] = []

    for unit in new_catalog["units"]:
        uid = unit["id"]
        for s_idx, span in enumerate(unit["spans"]):
            p_key = (span["doc_id"], span["page_number"])
            target_page = target_page_mapping.get(p_key)
            if target_page is None:
                raise ValueError(f"Target page not found for unit '{uid}' page {p_key}")

            old_quote = span["quote"]
            old_start = span["start_char"]
            old_end = span["end_char"]

            new_start, new_end = reanchor_span(
                quote=old_quote,
                page_text=target_page["text"],
                unit_id=uid,
                page_number=span["page_number"],
            )

            raw_slice = target_page["text"][new_start:new_end]
            new_quote = collapse_spaces(raw_slice)

            span["start_char"] = new_start
            span["end_char"] = new_end
            span["quote"] = new_quote

            old_quote_sha = hashlib.sha256(old_quote.encode("utf-8")).hexdigest()
            new_quote_sha = hashlib.sha256(new_quote.encode("utf-8")).hexdigest()
            quote_changed = old_quote != new_quote

            span_audit = {
                "unit_id": uid,
                "span_index": s_idx,
                "doc_id": span["doc_id"],
                "page_number": span["page_number"],
                "old_start_char": old_start,
                "old_end_char": old_end,
                "new_start_char": new_start,
                "new_end_char": new_end,
                "old_start": old_start,
                "old_end": old_end,
                "new_start": new_start,
                "new_end": new_end,
                "old_offsets": (old_start, old_end),
                "new_offsets": (new_start, new_end),
                "old_quote_sha256": old_quote_sha,
                "new_quote_sha256": new_quote_sha,
                "old_quote_sha": old_quote_sha,
                "new_quote_sha": new_quote_sha,
                "quote_changed": quote_changed,
                "normalization_rule": NORMALIZATION_RULE,
            }
            span_audits.append(span_audit)

            if quote_changed:
                diff_entry = {
                    "unit_id": uid,
                    "span_index": s_idx,
                    "page_number": span["page_number"],
                    "old_quote_sample": old_quote[:60],
                    "new_quote_sample": new_quote[:60],
                    "old_quote_sha256": old_quote_sha,
                    "new_quote_sha256": new_quote_sha,
                }
                if any(dash in old_quote for dash in TYPOGRAPHIC_DASHES):
                    diff_entry["glyph_match"] = (
                        "Finite typographic dash in source quote unified to ASCII '-' in target text"
                    )
                glyph_differences.append(diff_entry)

    # Validate new catalog after mapping
    validate_catalog(new_catalog, target_pages)

    # Compute raw SHA hashes
    sorted_source = sorted(source_pages, key=lambda p: (p["doc_id"], p["page_number"]))
    sorted_target = sorted(target_pages, key=lambda p: (p["doc_id"], p["page_number"]))

    src_raw_concat = "".join(p["text"] for p in sorted_source)
    tgt_raw_concat = "".join(p["text"] for p in sorted_target)
    src_raw_hash = hashlib.sha256(src_raw_concat.encode("utf-8")).hexdigest()
    tgt_raw_hash = hashlib.sha256(tgt_raw_concat.encode("utf-8")).hexdigest()

    src_pages_hashes = {
        f"{p['doc_id']}:{p['page_number']}": hashlib.sha256(p["text"].encode("utf-8")).hexdigest()
        for p in sorted_source
    }
    tgt_pages_hashes = {
        f"{p['doc_id']}:{p['page_number']}": hashlib.sha256(p["text"].encode("utf-8")).hexdigest()
        for p in sorted_target
    }

    orig_units = source_catalog.get("units", [])
    orig_unit_count = len(orig_units)
    new_unit_count = len(new_catalog.get("units", []))
    orig_span_count = sum(len(u.get("spans", [])) for u in orig_units)
    new_span_count = sum(len(u.get("spans", [])) for u in new_catalog.get("units", []))

    audit_report = {
        "source_canonical_raw_sha256": src_raw_hash,
        "target_canonical_raw_sha256": tgt_raw_hash,
        "source_canonical_raw_sha": src_raw_hash,
        "target_canonical_raw_sha": tgt_raw_hash,
        "source_raw_sha256": src_raw_hash,
        "target_raw_sha256": tgt_raw_hash,
        "source_pages_sha256": src_pages_hashes,
        "target_pages_sha256": tgt_pages_hashes,
        "original_unit_count": orig_unit_count,
        "new_unit_count": new_unit_count,
        "original_span_count": orig_span_count,
        "new_span_count": new_span_count,
        "normalization_rule": NORMALIZATION_RULE,
        "spans": span_audits,
        "glyph_differences": glyph_differences,
    }

    return new_catalog, audit_report
