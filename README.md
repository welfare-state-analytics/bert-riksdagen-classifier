# BERT Riksdagen classifier

Currently binary classification of segment types.

## Data

- A sample of 6(?) pages per year has been randomly sampled from the corpus. It is in _sample.csv_ and covers the era before documents were born digital.

- All pages have been annotated for each label (currently note/seg/intro). The annotated data is located in the labels respective directories.

- The data has been annotated such that if ANY information from a certain tag is present in the OCR box, then I label it as belonging to that tag. This is a way to avoid ambiguities and the use of more complicated heuristics. The only unclear situation is when intros overlap with segments, in which case I have labeled it as intro and seg. If the observation is only an intro, I believe I labeled it as intro and NOTE. This can be fixed very easily if some other solution seems better.

- _join_annotated_data.py_ is used to join all of the annotated data into a single csv file with the multiple labels dummy coded. Data is stored as _training_data.csv_. Example usage:

> python join_annotated_data.py seg note intro

## Classifier

As there likely is some additional work necessary to get a good multi label classifier in place, with good post processing, etc. I wrote a quick test script, _train_binary_bert.py_,  to create a binary note/seg classifier (where everything not labeled as seg is labeled note instead). The script works but is not completely finished, it for example does not do test set predictions atm.

## Notes

Everything in this folder should probably not be here in input. But I figured it should all be contained in the same folder for when passing it on.

## Going forward

Train a multi label BERT and handle overlapping labels in post.
Currently we only have 3 labels. The overlaps can initially be handled pretty simply using the rules:

- if intro --> tag as intro no matter what

- if note and seg --> label as seg

You can then label additional tags such as header or debate topic. It takes me ~2h to label all observations in the sample per label.
