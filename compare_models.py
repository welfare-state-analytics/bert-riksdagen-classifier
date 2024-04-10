from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd
import Levenshtein
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


def evaluate(model1, model2,tokenizer, loader, dataset ,label_names):
    loss1, accuracy1 = 0.0, []
    loss2, accuracy2 = 0.0, []
    model1.eval()
    model2.eval()
    true_labels, pred_labels1, pred_labels2, misclassified_examples = [], [], [], []
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(args.device)
        input_mask = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        output1 = model1(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        output2 = model2(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        loss1 += output1.loss.item()
        loss2 += output2.loss.item()
        preds_batch1 = torch.argmax(output1.logits, axis=1)
        preds_batch2 = torch.argmax(output2.logits, axis=1)
        batch_acc1 = torch.mean((preds_batch1 == labels).float())
        batch_acc2 = torch.mean((preds_batch2 == labels).float())
        accuracy1.append(batch_acc1)
        accuracy2.append(batch_acc2)
        true_labels.extend(labels.cpu().numpy())
        pred_labels1.extend(preds_batch1.cpu().numpy())
        pred_labels2.extend(preds_batch2.cpu().numpy())
        
        for true_label, pred_label1, pred_label2 , input_id  in zip(labels, preds_batch1, preds_batch2, input_ids):
            if true_label != pred_label1 or true_label != pred_label2 :
                text = tokenizer.decode(input_id, skip_special_tokens=True)
                matching_rows = dataset[dataset['content'].apply(lambda x: Levenshtein.ratio(text, x) >= 0.9)]
                if not matching_rows.empty:
                    github = matching_rows['github'].iloc[0]
                    protocol_id = matching_rows['protocol_id'].iloc[0]

                    misclassified_examples.append({'text': text, 'true_label': label_names[true_label.item()], 'predicted1':label_names[pred_label1.item()],'predicted2':label_names[pred_label2.item()], 'github': github, 'protocol_id': protocol_id})
                else:
                    print(f"no matching row for text: {text}")

    misclassified_df = pd.DataFrame(misclassified_examples)
    misclassified_df.to_csv('data/compare_misclassified_examples.csv', index=False, columns=['text', 'true_label', 'predicted1', 'predicted2', 'github', 'protocol_id'])
    

    # Print misclassified examples
    print("\nMisclassified Examples:")
    print(misclassified_examples)

    print("\nAccuracy model 1:", accuracy_score(true_labels, pred_labels1))
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels1, target_names=list(label_names)))

    
    print("\nAccuracy model 2:", accuracy_score(true_labels, pred_labels2))
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels2, target_names=list(label_names)))


def main(args):
    os.makedirs(args.model_filename1, exist_ok=True)
    logging.basicConfig(filename=f'{args.model_filename1}/evaluation.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    model1 = AutoModelForSequenceClassification.from_pretrained(
        args.model_filename1,
        num_labels=len(label_dict),
        id2label=label_dict).to(args.device)

    model2 = AutoModelForSequenceClassification.from_pretrained(
        args.model_filename2,
        num_labels=len(label_dict),
        id2label=label_dict).to(args.device)
    LOGGER.info("Load and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_filename1)

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
    evaluate(model1, model2, tokenizer, test_loader,df, label_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_filename1", type=str, default="trained/binary_note_seg_model", help="Save location for the model")
    parser.add_argument("--model_filename2", type=str, default="trained/binary_note_seg_model", help="Save location for the model")
    parser.add_argument("--base_model", type=str, default="KBLab/bert-base-swedish-cased", help="Base model that the model is initialized from")
    parser.add_argument("--tokenizer", type=str, default="KBLab/bert-base-swedish-cased", help="Which tokenizer to use; accepts local and huggingface tokenizers.")
    parser.add_argument("--label_names", type=str, nargs="+", default=None, help="A list of label names to be used in the classifier. If None, takes class names from 'tag' column in the data.")
    parser.add_argument("--data_path", type=str, default="data/pilot/val_processed_data.csv", help="Testing data as a .CSV file. Needs to have 'content' (X) and 'tag' (Y) columns.")
    parser.add_argument("--device", type=str, default="cuda", help="Which device to use for training. Use 'cpu' for CPU.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
