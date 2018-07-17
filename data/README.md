# SWAG dataset
Each item in the CSV is an example. It's conveniently in two formats. 


## full
In `train_full.csv` or `val_full.csv`: we have both the texts of the endings/contexts, as well as the ordinal labels (likely, unlikely), some identification `video-id`,`fold-ind`, the full context `startphrase` (also, split into `sent1`,`sent2`), and the endings. There's the gold ending `gold-ending` and its source `gold-source` which is either generated or comes from the found completion. There's an ordinal label for the gold as well `gold-type`. We also have 3-4 distractors `distractor-N` and an ordinal label for each one `distractor-N-type.` The reason it's 3 or 4 is that sometimes there were more answers filtered out as gibberish by the annotators. When there is a 4th distractor, it's often of lower quality than the others (ranked the most plausible).

## regular (shuffled)

This could be more interesting for modeling, and it's the way the test data is formatted. For each `startphrase` (also, split into `sent1`,`sent2`) we have 4 endings, and a label which says the correct one. 