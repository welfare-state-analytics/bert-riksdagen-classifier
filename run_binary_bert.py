import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
from transformers import get_linear_schedule_with_warmup
import argparse
from tqdm import tqdm
import os
import scipy.stats
from pathlib import Path
from lxml import etree

def encode(df):
    tokenizer = AutoTokenizer.from_pretrained('KBLab/bert-base-swedish-cased')

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        encoded_dict = tokenizer.encode_plus(
                            row['content'],                      
                            add_special_tokens = True,
                            max_length = 512,
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def predict(model, loader):
    preds = []
    entropies = []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(args.device)
        input_mask = batch[1].to(args.device)
        output = model(input_ids,
            token_type_ids=None, 
            attention_mask=input_mask)

        preds.extend(torch.argmax(output.logits, axis=1).tolist())
        probs = torch.sigmoid(output.logits.cpu().detach()).numpy()
        entropy = scipy.stats.entropy(probs, axis=1)
        entropies.extend(entropy)

    assert len(preds) == len(entropies)
    return preds, entropies


def main(args):
    rows = []
    parser = etree.XMLParser(remove_blank_text=True)
    for folder in tqdm(list(Path(args.corpus_path).glob("protocols/*"))):
        year = folder.stem[:4]
        year = int(year)
        if args.start <= year <= args.end:
            for file in folder.glob("*.xml"):
                with file.open() as f:
                    root = etree.parse(f, parser).getroot()

                for body in root.findall(".//{http://www.tei-c.org/ns/1.0}body"):
                    for div in body.findall("{http://www.tei-c.org/ns/1.0}div"):
                        for elem in div:
                            if elem.tag == "{http://www.tei-c.org/ns/1.0}u":
                                for subelem in elem:
                                    xml_id = subelem.attrib["{http://www.w3.org/XML/1998/namespace}id"]
                                    content = " ".join(subelem.text.strip().split())
                                    rows.append([xml_id, content])
                            elif elem.tag != "{http://www.tei-c.org/ns/1.0}pb":
                                xml_id = elem.attrib["{http://www.w3.org/XML/1998/namespace}id"]
                                content = " ".join(elem.text.strip().split())
                                rows.append([xml_id, content])

    # TODO: read in data
    df = pd.DataFrame(rows, columns=["id", "content"])
    print(df)
    # Preprocess datasets
    print("Encode...")
    input_ids, attention_masks = encode(df)
    print("Generate dataset...")
    valid_dataset = TensorDataset(input_ids, attention_masks)

    print("Generate dataloader...")
    valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size = args.batch_size,
            num_workers = args.num_workers
        )

    print("Load model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'KBLab/bert-base-swedish-cased',
        num_labels=2)
    checkpoint = torch.load(args.model_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    valid_losses = []
    best_valid_loss = float('inf')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Predict...")
    # Evaluation
    preds, entropies = predict(model, valid_loader)
    #print(preds)
    #print(entropies)

    df["preds"] = preds
    print(df)

    result_df = df[["id", "preds"]]
    result_df.to_csv(f"preds-{args.start}-{args.end}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus_path", type=str, default="../riksdagen-corpus/corpus/")
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2021)
    parser.add_argument("--model_filename", type=str, default="trained/binary_note_seg_model.pth")
    parser.add_argument("--data_folder", type=str, default="data/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
