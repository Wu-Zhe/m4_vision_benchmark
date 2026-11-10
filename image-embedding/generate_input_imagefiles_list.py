#!/usr/bin/env python3
"""
Parse an eBay item/image input file and write one record per image URL: <item_id>_<seq>.webp

Rules:
  * field 1 = item id (the base file name), field 2 = ';'-separated image URLs
  * the 1st URL gives <item_id>_1.webp, the 2nd gives <item_id>_2.webp, and so on:
    every URL produces exactly one record, numbered by its position in the list
    (empty entries such as a trailing ';' are ignored and don't use up a number)
  * the record name depends only on the item id and the position. The eBay image id inside the
    URL plays no part in it; it is only used for the webp URL column of --with-url
  * default delimiter = first run of whitespace after the item id
  * blank lines, header lines (no "http" in field 2) and lines with < 2 fields are skipped
  * the item id is used as-is, except that any char outside [A-Za-z0-9._-] becomes '_'
    (the same rule as the Java downloader, so names match its files)

The input may be plain text or gzip-compressed. Gzip is detected from the file's first two
bytes (magic number 1f 8b), not from the extension, so "items.txt.gz" and a gzip file that was
renamed to "items.txt" both work.

Usage:
  python generate_filenames.py sample-input.txt                  # writes image_records.txt
  python generate_filenames.py sample-input.txt.gz               # gzipped input, same options
  python generate_filenames.py sample-input.txt -o records.txt
  python generate_filenames.py sample-input.txt --with-url       # "<file>\\t<webp url>" per record
  python generate_filenames.py input.csv --delimiter comma --size 500
"""
import argparse
import gzip
import re
import sys
import zlib

# Matches .../images/g/<id>/s-l225.jpg and .../thumbs/images/g/<id>/...
ID_IN_G = re.compile(r"/images/g/([^/?#]+)/")
# Matches .../00/s/<base64>/z/<id>/$_12.JPG
ID_IN_Z = re.compile(r"/z/([^/?#]+)/")

DELIMITERS = {"tab": "\t", "\\t": "\t", "comma": ",", "pipe": "|", "semicolon": ";"}


def open_text(path):
    """Open path as UTF-8 text, transparently decompressing gzip (detected by magic bytes).

    utf-8-sig strips a leading BOM if present. Multi-member gzip files are read in full.
    """
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return open(path, encoding="utf-8-sig")


def extract_image_id(url):
    """Return the eBay image id found in url, or None."""
    m = ID_IN_G.search(url) or ID_IN_Z.search(url)
    return m.group(1) if m else None


def to_webp_url(image_id, size=640, base="https://i.ebayimg.com"):
    return f"{base.rstrip('/')}/images/g/{image_id}/s-l{size}.webp"


def parse_row(line, delimiter=None):
    """Return (item_id, [raw_urls]) for a data row, or None if the row should be skipped."""
    if not line.strip():
        return None
    if delimiter is None:
        fields = line.strip().split(None, 1)          # split at first run of whitespace
    else:
        fields = line.rstrip("\r\n").split(delimiter)
    if len(fields) < 2:
        print(f"warning: fewer than 2 fields, skipped: {line.strip()[:60]!r}", file=sys.stderr)
        return None
    item_id = fields[0].strip()
    tokens = fields[1].strip().split()                 # URLs contain no whitespace
    url_field = tokens[0] if tokens else ""
    if not item_id or "http" not in url_field:
        return None                                    # header row or no images
    return item_id, url_field.split(";")


def generate_records(lines, delimiter=None, size=640, stats=None):
    """Yield (file_name, webp_url) for every image URL in the input lines.

    There is one record per URL. webp_url is "" when no image id can be found in the URL;
    the file name is unaffected by that.
    """
    stats = stats if stats is not None else {}
    for line in lines:
        parsed = parse_row(line, delimiter)
        if parsed is None:
            stats["rows_skipped"] = stats.get("rows_skipped", 0) + 1
            continue
        stats["rows"] = stats.get("rows", 0) + 1
        item_id, raw_urls = parsed
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", item_id)

        seq = 0
        for raw in raw_urls:
            url = raw.strip()
            if not url:
                continue
            seq += 1                                   # position in the URL list, starting at 1
            image_id = extract_image_id(url)
            if image_id is None:
                stats["urls_without_image_id"] = stats.get("urls_without_image_id", 0) + 1
                webp_url = ""
            else:
                webp_url = to_webp_url(image_id, size)
            yield f"{safe_id}_{seq}.webp", webp_url


def main():
    ap = argparse.ArgumentParser(description="Generate <item_id>_<seq>.webp records from an eBay input file.")
    ap.add_argument("input", help="input text file (item id + ';'-separated image URLs)")
    ap.add_argument("-o", "--output", default="image_records.txt", help="output file (default: image_records.txt)")
    ap.add_argument("--delimiter", help="tab|comma|pipe|semicolon|<char> (default: first run of whitespace)")
    ap.add_argument("--size", type=int, default=640, help="s-l size used with --with-url (default 640)")
    ap.add_argument("--with-url", action="store_true",
                    help="write '<file_name><TAB><webp url>' instead of just the file name")
    args = ap.parse_args()

    delimiter = DELIMITERS.get(args.delimiter.lower(), args.delimiter) if args.delimiter else None
    stats = {}
    count = 0
    try:
        with open_text(args.input) as fin, open(args.output, "w", encoding="utf-8", newline="\n") as fout:
            for name, url in generate_records(fin, delimiter, args.size, stats):
                fout.write(f"{name}\t{url}\n" if args.with_url else f"{name}\n")
                count += 1
    except (OSError, EOFError, zlib.error, UnicodeDecodeError) as e:
        # OSError covers missing files and gzip.BadGzipFile; EOFError = truncated gzip;
        # zlib.error = corrupt compressed data
        print(f"error: cannot read {args.input}: {e}", file=sys.stderr)
        print(f"       {args.output} may contain partial results ({count} records)", file=sys.stderr)
        sys.exit(1)

    msg = (f"{count} records from {stats.get('rows', 0)} rows written to {args.output} "
           f"({stats.get('rows_skipped', 0)} rows skipped)")
    print(msg, file=sys.stderr)
    if args.with_url and stats.get("urls_without_image_id"):
        print(f"note: {stats['urls_without_image_id']} URL(s) had no recognizable image id; "
              f"their webp URL column is empty", file=sys.stderr)


if __name__ == "__main__":
    main()
