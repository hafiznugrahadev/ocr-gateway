from app.utils.errors import OcrGatewayError


def parse_pages(spec: str | None, total: int) -> list[int]:
    """
    Parse a pages selector spec into a sorted list of 1-based page numbers.

    Examples:
      "all"       -> [1..total]
      "1"         -> [1]
      "1-5"       -> [1,2,3,4,5]
      "1,3,5"     -> [1,3,5]
      "1-3,7,9-10"-> [1,2,3,7,9,10]

    Pages out of range (>total or <1) raise INVALID_PAGES.
    """
    if total <= 0:
        return []

    raw = (spec or "all").strip().lower()
    if raw in ("", "all"):
        return list(range(1, total + 1))

    selected: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            try:
                start_s, end_s = chunk.split("-", 1)
                start, end = int(start_s), int(end_s)
            except ValueError as exc:
                raise OcrGatewayError(
                    error_code="INVALID_PAGES",
                    detail=f"Invalid page range: {chunk!r}",
                    status_code=400,
                ) from exc
            if start < 1 or end < start:
                raise OcrGatewayError(
                    error_code="INVALID_PAGES",
                    detail=f"Invalid page range: {chunk!r}",
                    status_code=400,
                )
            for p in range(start, end + 1):
                if p > total:
                    raise OcrGatewayError(
                        error_code="INVALID_PAGES",
                        detail=f"Page {p} out of range (document has {total} pages)",
                        status_code=400,
                    )
                selected.add(p)
        else:
            try:
                p = int(chunk)
            except ValueError as exc:
                raise OcrGatewayError(
                    error_code="INVALID_PAGES",
                    detail=f"Invalid page number: {chunk!r}",
                    status_code=400,
                ) from exc
            if p < 1 or p > total:
                raise OcrGatewayError(
                    error_code="INVALID_PAGES",
                    detail=f"Page {p} out of range (document has {total} pages)",
                    status_code=400,
                )
            selected.add(p)

    return sorted(selected)
