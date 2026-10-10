"""Temporal-holdout priors: CollecTRI edges dated by their earliest cited PubMed record and split at a cutoff year.

The prior holds the edges first cited up to the cutoff. Labels mark the edges first cited after it and the
undated edges, which the evaluation excludes. ``--check`` rebuilds every setting and compares it with the
arrays saved in the data bundle.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from beacon.data import Bundle
from preparation.prior import training

CUTOFFS = (2000, 2005, 2010)
CONTEXTS = ("k562", "tcell_resting", "tcell_stimulated", "rpe1")


def references(bundle):
    """CollecTRI rows with their dated PubMed identifiers, and the year of each identifier."""
    section = bundle.manifest["temporal"]["references"]
    years = bundle.frame(section["pmid_years"], sep="\t", dtype={"pmid": str})
    year = dict(zip(years.pmid, years.year.astype(int)))
    table = bundle.frame(section["collectri"], sep="\t", low_memory=False)
    table["pmids"] = table.references.fillna("").astype(str).map(lambda r: sorted({m for m in re.findall(r":(\d+)", r) if m in year}))
    return table, year


def dated_edges(table, year):
    """Each TF-target edge with the earliest year among its dated references (NaN when none is dated)."""
    first = table.pmids.map(lambda pmids: min((year[m] for m in pmids), default=np.nan))
    return table.assign(first_year=first).groupby(["source_genesymbol", "target_genesymbol"], as_index=False).first_year.min()


def setting(bundle, context, cutoff, edges):
    """Expression arrays, training prior, candidate mask and later-edge labels for one context and cutoff."""
    data = bundle.arrays(bundle.manifest["prepared"][context]["expression"])
    train, discovery, _ = training(data["expression"], data["genes"], data["symbols"], data["tf_indices"], edges[edges.first_year <= cutoff])
    symbols = data["symbols"]
    counts = pd.Series(symbols).value_counts()
    index = {s: i for i, s in enumerate(symbols) if counts[s] == 1}

    def matrix(frame):
        values = np.zeros((len(symbols), len(symbols)), dtype=bool)
        for source, target in zip(frame.source_genesymbol, frame.target_genesymbol):
            if source in index and target in index and index[source] != index[target]:
                values[index[source], index[target]] = True
        return values[train["tf_indices"]]

    labels = dict(later=matrix(edges[edges.first_year > cutoff]), undated=matrix(edges[edges.first_year.isna()]))
    return data, train, discovery, labels


def saved(bundle, context, cutoff):
    entry, = [e for e in bundle.manifest["temporal"]["settings"] if e["context"] == context and e["cutoff"] == cutoff]
    return entry


def check(bundle):
    edges = dated_edges(*references(bundle))
    for cutoff in CUTOFFS:
        for context in CONTEXTS:
            _, train, discovery, labels = setting(bundle, context, cutoff, edges)
            entry = saved(bundle, context, cutoff)
            stored = bundle.arrays(entry["training"])
            same = (all(np.array_equal(train[k], stored[k]) for k in ("train", "genes", "symbols", "tf_indices"))
                    and np.array_equal(discovery, bundle.array(entry["discovery"]))
                    and all(np.array_equal(labels[k], bundle.arrays(entry["labels"])[k]) for k in ("later", "undated")))
            print(cutoff, context, "identical" if same else "DIFFERENT", flush=True)
            if not same:
                raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args()
    check(Bundle(args.config))


if __name__ == "__main__":
    main()
