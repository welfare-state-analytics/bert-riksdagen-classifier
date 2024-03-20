"""
Some of the annotations are missing the full text of the paragraph.
Fetch the full text in the annotations from the riksdagen corpus.
"""
import argparse
from pyriksdagen.utils import protocol_iterators, infer_metadata, elem_iter, XML_NS
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from lxml import etree
import re
import pandas as pd

def main(args):
    df = pd.read_csv(args.df)
    print(df)
    protocols = sorted(list(protocol_iterators(args.records_folder)))
    parser = etree.XMLParser(remove_blank_text=True)

    annotated_protocols = set(list(df["protocol_id"]))
    element_texts = []

    for protocol in tqdm(protocols, total=len(protocols)):
        metadata = infer_metadata(protocol)
        protocol_id = metadata["protocol"].replace("_", "-")
        if protocol_id in annotated_protocols:
            with open(protocol) as f:
                root = etree.parse(f, parser).getroot()
            for tag, elem in elem_iter(root):
                if elem.text is not None:
                    text = " ".join(elem.text.split())
                    elem_id = elem.attrib[f"{XML_NS}id"]
                    element_texts.append([elem_id, text])
                elif tag == "u":
                    for subelem in elem:
                        if subelem.text is not None:
                            text = " ".join(subelem.text.split())
                            subelem_id = subelem.attrib[f"{XML_NS}id"]
                            element_texts.append([subelem_id, text])

    element_texts = pd.DataFrame(element_texts, columns=["elem_id", "full_text"])
    len_before = len(df)

    # Merge while preserving order
    df["index"] = list(range(len(df)))
    df = df.merge(element_texts, on="elem_id", how="left")
    df = df.sort_values("index")
    columns = [col for col in df.columns if col not in ["full_text", "index"]]
    df = df[columns]

    assert len_before == len(df), f"Len before {len_before} len now {len(df)}"

    df.to_csv(args.df, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records_folder", type=str, default="corpus/records")
    parser.add_argument("--df", type=str, default="data/raw/train.csv")
    args = parser.parse_args()
    main(args)
