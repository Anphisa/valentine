# Schema matcher tests based on Valentine
For my master's thesis on schema integration, I did some tests on schema matching algorithms.
These are based on Valentine (https://github.com/delftdata/valentine).


Three subfolders with data needed:
* data: https://archive.org/details/data.7z_202303 (data hand-labelled by me)
* Valentine-datasets (datasets used in Valentine evaluation, documented here: https://delftdata.github.io/valentine/ (Zenodo link for data))
* Valentine-output (output from Valentine evaluation, documented here: https://github.com/delftdata/valentine-paper-results (subfolder outputs for data))
Download data, extract it, and place it in subfolders.

Valentine was extended with JaccardLevenshtein matcher based on column names only (https://github.com/Anphisa/valentine/tree/master/valentine/algorithms/jaccard_levenshtein_colnames_only).
