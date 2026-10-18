import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from beacon.data import configuration

QUERY = ("https://omnipathdb.org/interactions?datasets=dorothea&dorothea_levels=A%2CB&genesymbols=yes"
         "&fields=sources%2Creferences%2Cdorothea_level&organisms=9606")
LEVELS = {"A", "B"}
EXCLUDED_SOURCES = "ReMap|ENCODE"
NATIVE_REFERENCE = re.compile(r"(?:^|;)DoRothEA:(\d+)")
BEFORE_YEAR = 2022


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def allowed_pmids(dates):
    return {pmid for pmid, record in dates.items()
            if re.match(r"^\d{4}", record.get("sortpubdate") or "") and int(record["sortpubdate"][:4]) < BEFORE_YEAR}


def build(table, dates):
    counts = {"download_rows": len(table)}
    table = table[table.dorothea_level.map(lambda level: bool(LEVELS & set(level.split(";"))))].copy()
    counts["level_a_or_b_rows"] = len(table)
    table["dorothea_pmids"] = table.references.map(lambda refs: ";".join(sorted(set(NATIVE_REFERENCE.findall(refs)))))
    table = table[~table.sources.str.contains(EXCLUDED_SOURCES, case=False, regex=True) & table.dorothea_pmids.ne("")].copy()
    counts["without_remap_or_encode_with_native_pmid_rows"] = len(table)
    native = set(";".join(table.dorothea_pmids).split(";")) - {""}
    allowed = allowed_pmids(dates) & native
    counts.update(native_pmids=len(native), undated_native_pmids=len(native - set(dates)), allowed_pmids=len(allowed))
    table["allowed_pmids"] = table.dorothea_pmids.map(lambda pmids: ";".join(p for p in pmids.split(";") if p in allowed))
    table = table[table.allowed_pmids.ne("")]
    counts["allowed_prior_rows"] = len(table)
    return table, counts


def render(table):
    return table.to_csv(sep="\t", index=False, lineterminator="\n").encode()


def compare(built, path):
    rebuilt, existing = render(built), path.read_bytes()
    print("rebuilt sha256", hashlib.sha256(rebuilt).hexdigest(), "existing sha256", hashlib.sha256(existing).hexdigest())
    if rebuilt == existing:
        print("byte identical:", len(built), "rows")
        return True
    frozen = read(path)
    if list(frozen.columns) != list(built.columns):
        print("columns differ: rebuilt", list(built.columns), "existing", list(frozen.columns))
    columns = [c for c in built.columns if c in frozen.columns]
    new = list(built[columns].itertuples(index=False, name=None))
    old = list(frozen[columns].itertuples(index=False, name=None))
    print("rows: rebuilt", len(new), "existing", len(old), "| same set", set(new) == set(old), "| same order", new == old)
    for label, rows in (("only rebuilt", set(new) - set(old)), ("only existing", set(old) - set(new))):
        for row in sorted(rows):
            print(label, dict(zip(columns, row)))
    if set(new) == set(old) and new != old:
        first = next(i for i, (a, b) in enumerate(zip(new, old)) if a != b)
        print("first order difference at data row", first + 1)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dorothea", type=Path, help="OmniPath DoRothEA download (default raw/prior/dorothea.tsv)")
    parser.add_argument("--dates", type=Path, help="PubMed summaries (default raw/prior/pubmed_dates.json)")
    parser.add_argument("--output", type=Path, help="prior table (default raw/prior/allowed_prior.tsv)")
    parser.add_argument("--check", action="store_true", help="rebuild in memory and compare with the existing output")
    args = parser.parse_args()
    prior = configuration(args.config)["data_root"] / "raw/prior"
    dorothea = args.dorothea or prior / "dorothea.tsv"
    dates = args.dates or prior / "pubmed_dates.json"
    output = args.output or prior / "allowed_prior.tsv"
    table, counts = build(read(dorothea), json.loads(dates.read_text()))
    print(json.dumps(counts, indent=2), flush=True)
    if args.check:
        if not compare(table, output):
            raise SystemExit(1)
        return
    with output.open("xb") as handle:
        handle.write(render(table))


if __name__ == "__main__":
    main()
