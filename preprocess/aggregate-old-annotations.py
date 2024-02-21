"""
Aggregate old .txt annotations into a JSON format
"""
import json
import yaml
import argparse
from pathlib import Path
from trainerlog import get_logger

LOGGER = get_logger("aggregate")
def main(args):
    l = []
    for folder in args.folders:
        folder = Path(folder)
        files = folder.glob("*")
        d = {}
        for file in sorted(files):
            if file.suffix in [".txt"] and file.stem in ["original", "annotated"]:
                with file.open() as f:
                    text = f.read()

                if file.stem == "annotated":
                    text = text.replace("ENDSPEECH", "")
                    text = text.replace("ENDDESCRIPTION", "")
                    text = text.replace("CONTINUE", "BEGIN")
                    textlist = text.split("BEGIN")
                    def classify(t):
                        t = t.strip()
                        if t[:6] == "SPEECH":
                            t0 = {"class": "u", "p": t[6:]}
                        elif t[:11] == "DESCRIPTION":
                            t0 = {"class": "note", "p": t[11:]}
                        else:
                            t0 = {"class": "note", "p": t[11:]}

                        if t0["p"] != "":
                            return t0

                    text = [classify(t) for t in textlist if classify(t) is not None]

                d[file.stem] = text
            elif file.suffix in [".yml", ".yaml"]:
                with file.open() as f:
                    info = yaml.safe_load(f)
                    d.update(info)

        l.append(d)
    txt = json.dumps(l, ensure_ascii=False, indent=2)
    with open(args.outpath, "w") as f:
        f.write(txt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", type=str, nargs="+", default=[], help="Folders to be aggregated")
    parser.add_argument("--outpath", type=str, default="data/old-curation.json", help="Output file")
    args = parser.parse_args()
    main(args)
