# SQuALITY

This repo contains the [SQuALITY](TODO) (Summarization-format QUestion Answering with Long Input Texts, Yes!) dataset and supporting code.
SQuALITY is a question-focused, long-document, multi-reference summarization dataset.
The source documents are short stories from Project Gutenberg on the order of 4000--6000 words long.
The stories are split such that stories in this dataset that also appear in the [QuALITY](https://arxiv.org/abs/2112.08608) dataset are assigned to the same split.
Each story is paired with a set of five questions, the first of which is always "What is the plot of the story?"
Each question has four reference summaries, all of which are written by writers from Upwork and NYU undergraduates who consented to having their writing distributed for research purposes.

# Data and Format

The dataset lives in `data`.
There are currently two versions of the dataset:
* `v1` consists of 100 stories and was the version of the dataset used in the initial version of the paper.
* `v2` consists of 125 stories and is a superset of `v1`. It maintains the same split assignments for stories that appear in both versions of the dataset.

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

# Citation

TBA.

# Contact

Open an issue on this repo or email wangalexc _at_ gmail.com
