"""Render manuscript figures and table bodies from the configured data bundle."""
from pathlib import Path
import argparse
from figures.build import Build

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=("main","supplementary","tables","sensitivity","checks"))
    p.add_argument("--only",help="Descriptive output stem")
    p.add_argument("--config",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args();build=Build(a.config,a.out,a.only)
    if a.kind=="checks":
        import hashlib
        from figures.checks import run
        run(build.bundle,build.directory)
        for path in sorted(build.directory.glob("*.csv")):
            build.outputs.append(dict(output=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        build.finish()
        return
    if a.kind=="sensitivity":
        from figures.sensitivity import generate
        generate(build,build.directory)
        build.finish()
        return
    build.validate_labels()
    metrics,completion=build.pools()
    from figures import render,tables
    if a.kind=="main":render.main_figures(build,completion)
    elif a.kind=="supplementary":render.supplementary(build,metrics,completion)
    else:
        tables.dimensions(build);tables.trrust(build);tables.scregnet(build,completion)
        tables.expression(build,metrics);tables.sensitivity(build);tables.response_tables(build)
    build.finish()

if __name__=="__main__":main()
