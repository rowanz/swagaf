# SWAG dataset
Each item in the CSV is an example. It's conveniently in two formats. 


## full
In `train_full.csv` or `val_full.csv`: we have both the texts of the endings/contexts, as well as the ordinal labels (likely, unlikely), some identification `video-id`,`fold-ind`, the full context `startphrase` (also, split into `sent1`,`sent2`), and the endings. There's the gold ending `gold-ending` and its source `gold-source` which is either generated or comes from the found completion. There's an ordinal label for the gold as well `gold-type`. We also have 3-4 distractors `distractor-N` and an ordinal label for each one `distractor-N-type.` The reason it's 3 or 4 is that sometimes there were more answers filtered out as gibberish by the annotators. When there is a 4th distractor, it's often of lower quality than the others (ranked the most plausible).


## regular (shuffled)

This could be more interesting for modeling, and it's the way the test data is formatted. For each `startphrase` (also, split into `sent1`,`sent2`) we have 4 endings, and a label which says the correct one. You can use `test.csv` for submission on the leaderboard here: [https://leaderboard.dev.allenai.org/swag/submission/create](https://leaderboard.dev.allenai.org/swag/submission/create). The fields are exactly the same as `val.csv` and `train.csv` except for the label.



## More info about gold-source
If the source starts with `gold`, it comes from the found data (from an actual video caption). This is the case for all questions in the val and test sets.
* `gold0-orig`: It was selected as the *best* answer by a turker.
* `gold1-orig`: It was selected as the *second best* answer by a turker.
* `gold0-reannot`: It was originally not selected as within the top two, so it was reannotated (with different negatives) and ranked as the best.
* `gold1-reannot`: It was originally not selected as within the top two, so it was reannotated (with different negatives) and ranked as second best.
* `gold0-reannot`: It was originally not selected as within the top two, so it was reannotated (with different negatives) and ranked as the best.

For training, we also have questions marked as `gen-orig`: these are generated answers that are selected as the *best* answer, while the real answer was selected as the second best (`gold1-orig`)

tl;dr you probably don't have to worry about this one. However, during training, some models work better if also shown the `gen-orig` examples, and some work better if not.