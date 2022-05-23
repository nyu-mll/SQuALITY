# SQuALITY

This repo contains the [SQuALITY](https://w4ngatang.github.io/static/papers/squality.pdf) (Summarization-format QUestion Answering with Long Input Texts, Yes!) dataset and supporting code.
SQuALITY is a question-focused, long-document, multi-reference summarization dataset.
The source documents are short stories from Project Gutenberg on the order of 4000-6000 words long.
The stories are split such that stories in this dataset that also appear in the [QuALITY](https://arxiv.org/abs/2112.08608) dataset are assigned to the same split.
Each story is paired with a set of five questions, the first of which is always "What is the plot of the story?"
Each question has four reference summaries, all of which are written by writers from Upwork and NYU undergraduates who consented to having their writing distributed for research purposes.

# Authors

Alex Wang, Richard Yuanzhe Pang, Angelica Chen, Jason Phang, Samuel R. Bowman

# Data and Format

The dataset lives in `data`.
There are currently two versions of the dataset:
* `v1` consists of 100 (split 39/25/36 between train/dev/test) stories and was the version of the dataset used in the initial version of the paper.
* `v2` consists of 127 (split 50/25/52) stories and is a superset of `v1`. It maintains the same split assignments for stories that appear in both versions of the dataset.

Each data file (`{train/dev/test}.jsonl`) is formatted as a JSON lines file.
Each row in the data file is a JSON dictionary with the following fields:
* metadata: the Gutenberg story ID, an internal UID, and the Project Gutenberg license
* document: the Gutenberg story
* questions: a list of questions and accompanying responses
    * question text
    * question number: the order in which that question was answered by the writers
    * responses: list of worker's response, where each response is a dictionary containing the (anonymized) worker ID, an internal UID, and their response to the question

# Baselines

TBA.

# License

The stories are distributed under the [Project Gutenberg license](https://www.gutenberg.org/policy/license.html) and the summaries are distributed under a [CC BY](https://creativecommons.org/licenses/by/4.0/) license, in `data/LICENSE`.

# Acknowledgements

This project has benefited from financial support to SB by Eric and Wendy Schmidt (made by recommendation of the Schmidt Futures program) and Apple, and from in-kind support by the NYU High-Performance Computing Center and Google Cloud.
This material is based upon work supported by the National Science Foundation under Grant Nos. 1922658 and 2046556. 
Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation. 

# Citation

TBA.

# Contact

Open an issue on this repo or email wangalexc _at_ gmail.com
