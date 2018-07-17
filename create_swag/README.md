# create_swag

this folder contains the scripts used to create SWAG, including adversarial filtering. Here's the rough overview:

1. Compile a bunch of datasets. We used MPII and ActivityNet Captions. 
2. Train the LM on those datasets (train first on toronto books). See the folder `lm/` for more info.
3. Oversample and then perform Adversarial Filtering. See `generate_candidates/`
4. Ask turkers to rank the distractors. You can use `turktemplate.html` as a starting point.
5. You're done!
 
### Important note:
This code is pretty hacky and comes with few guarantees (as with adversarial filtering itself); I figure you're probably going to need to do something different anyways. But hopefully it helps! Open up an issue if you notice anything wrong.