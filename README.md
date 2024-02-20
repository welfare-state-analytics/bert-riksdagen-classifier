# BERT Riksdagen classifier

Train a BERT-based classifier eg. for paragraph classification. The script saves the model in a huggingface pipeline compatible format. More information can be obtained by running

```sh
python3 train_binary_bert.py --help
```

Dependencies:

```
pandas, torch, transformers, tqdm, bidict, trainerlog
```