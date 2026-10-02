"""Download and verify the published bundle configured in config.toml."""
from pathlib import Path
import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from beacon.data import configuration, sha256

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path);a=p.parse_args()
    config=configuration(a.config);url=config.get("bundle_url","");expected=config.get("bundle_sha256","")
    if not url or len(expected)!=64:raise ValueError("Set bundle_url and bundle_sha256 in config.toml")
    destination=config["data_root"]
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="beacon_download_",dir=destination.parent) as temp:
        temp=Path(temp);archive=temp/"bundle.tar.gz";staged=temp/"contents";staged.mkdir()
        with urllib.request.urlopen(url) as response,archive.open("xb") as output:shutil.copyfileobj(response,output)
        if sha256(archive)!=expected:raise ValueError("Downloaded archive checksum differs")
        with tarfile.open(archive,"r:*") as bundle:
            for member in bundle.getmembers():
                target=(staged/member.name).resolve()
                if not target.is_relative_to(staged) or not (member.isfile() or member.isdir()):raise ValueError("Unsafe archive member")
            bundle.extractall(staged)
        if not (staged/"manifest.json").is_file():raise ValueError("Archive must contain manifest.json at its root")
        import json
        manifest=json.loads((staged/"manifest.json").read_text())
        if manifest.get("status")!="complete" or manifest.get("layout")!="plain_files":raise ValueError("Expected a complete plain-file bundle")
        names=[item["path"] for item in manifest["assets"]]
        if len(names)!=len(set(names)):raise ValueError("Duplicate bundle paths")
        if {str(path.relative_to(staged)) for path in staged.rglob("*") if path.is_file()} != {"manifest.json",*names}:raise ValueError("Archive files differ from manifest")
        for item in manifest["assets"]:
            name,digest=item["path"],item["sha256"]
            path=(staged/name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(staged) or sha256(path)!=digest:raise ValueError("Bundle file checksum differs")
        if destination.exists():raise FileExistsError(destination)
        staged.rename(destination)
    print(destination)
if __name__=="__main__":main()
