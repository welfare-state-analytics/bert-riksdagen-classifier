from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd
import argparse
import logging
import torch
import os
from bidict import bidict

LOGGER = logging.getLogger(__name__)

def encode(df, tokenizer):
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for ix, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
            row['content'],
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['tag'].tolist())

    return input_ids, attention_masks, labels


def evaluate(model, loader):
    loss, accuracy = 0.0, []
    model.eval()
    true_labels, pred_labels = [], []
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(args.device)
        input_mask = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        output = model(input_ids,
                       token_type_ids=None, 
                       attention_mask=input_mask, 
                       labels=labels)
        loss += output.loss.item()
        preds_batch = torch.argmax(output.logits, axis=1)
        batch_acc = torch.mean((preds_batch == labels).float())
        accuracy.append(batch_acc)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds_batch.cpu().numpy())
    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))


def main(args):
    os.makedirs(args.model_filename, exist_ok=True)
    logging.basicConfig(filename=f'{args.model_filename}/evaluation.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Create binary label where seg = 1
    df = df[df["content"].notnull()]
    label_names = args.label_names
    if label_names is None:
        label_names = sorted(list(set(df["tag"])))
    label_dict = {ix: name for ix, name in enumerate(label_names)}
    df["tag"] = [bidict(label_dict).inv[tag] for tag in df["tag"]]

    LOGGER.debug("load model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_filename,
        num_labels=len(label_dict),
        id2label=label_dict).to(args.device)

    LOGGER.info("Load and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_filename)

    LOGGER.info("Preprocess datasets...")
    input_ids, attention_masks, labels = encode(df, tokenizer)

    LOGGER.info(f"Labels: {labels}")

    dataset = TensorDataset(input_ids, attention_masks, labels)
 
    test_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    evaluate(model, test_loader)
    print("Classes: ", label_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_filename", type=str, default="trained/binary_note_seg_model", help="Save location for the model")
    parser.add_argument("--base_model", type=str, default="KBLab/bert-base-swedish-cased", help="Base model that the model is initialized from")
    parser.add_argument("--tokenizer", type=str, default="KBLab/bert-base-swedish-cased", help="Which tokenizer to use; accepts local and huggingface tokenizers.")
    parser.add_argument("--label_names", type=str, nargs="+", default=None, help="A list of label names to be used in the classifier. If None, takes class names from 'tag' column in the data.")
    parser.add_argument("--data_path", type=str, default="data/pilot/val_processed_data.csv", help="Testing data as a .CSV file. Needs to have 'content' (X) and 'tag' (Y) columns.")
    parser.add_argument("--device", type=str, default="cuda", help="Which device to use for training. Use 'cpu' for CPU.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
