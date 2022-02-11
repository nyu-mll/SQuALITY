# qfs-dataset

This repo contains the current training, dev, and test data for the (currently nameless) question-focused summarization dataset.
The source documents are short stories from Project Gutenberg on the order of 5000-6000 words long.
Each story is paired with a set of five questions, the first of which is always "What is the plot of the story?"
Each question has four reference summaries, all of which are written by writers from Upwork who consented to having their writing distributed for research purposes.

The stories are split such that stories in this dataset that also appear in the [QuALITY](https://arxiv.org/abs/2112.08608) dataset are assigned to the same split.

# Format

Each data file (`{train/dev/test}.jsonl`) is formatted as a JSON lines file.
Each row in the data file is a JSON dictionary with the following fields:
* metadata: the Gutenberg story ID, an internal UID, and the Project Gutenberg license
* document: the Gutenberg story
* questions: a list of questions and accompanying responses
    * question text
    * question number: the order in which that question was answered by the writers
    * responses: list of worker's response, where each response is a dictionary containing the worker ID, an internal UID, and their response to the question

# Baselines

# License

The stories are distributed under the [Project Gutenberg license](https://www.gutenberg.org/policy/license.html) and the crowdsourced writing is distributed under an MIT license.
