#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import os
import sys
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    LEDForConditionalGeneration
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    verbose: bool = field(
        default=False,
        metadata={
            "help": "Print info messages from inside model."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default='document',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    question_column: Optional[str] = field(
        default='questions',
        metadata={"help": "The name of the column in the datasets containing the question (for query-driven summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    prediction_out_file: Optional[str] = field(
        default="generated_predictions.txt",
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    include_info_in_prediction_file: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If true, include reference and part of input in the prediction file and save results as a jsonl file." "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_stories: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or learning curves with QFS, truncate the number of training *stories* to this "
            "value if set."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"), # nb: persian
    "wiki_asp": ("inputs", "targets"),
    "ccdv/arxiv-summarization": ("article", "abstract"),
    "ccdv/govreport-summarization": ("report", "summary"),
    "qds": ("story", "summary1"), # general plot summary
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sanity checks.
    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        #format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    print("loading data")
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name,
                                cache_dir=model_args.cache_dir)

    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.verbose = model_args.verbose
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    sep_tok = tokenizer.sep_token if tokenizer.sep_token is not None else ""
    sep_tok_id = tokenizer.sep_token_id

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # hack
    #model.config.gradient_checkpointing = True # for saving memory
    #model.config.use_cache = False             # for saving memory
    model.config.early_stopping = True
    model.config.no_repeat_ngram_story = 3
    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.question_column is not None:
        question_column = data_args.question_column
        if question_column not in column_names:
            raise ValueError(
                f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
            )
    else:
        question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        if isinstance(model, LEDForConditionalGeneration):
            # from https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing
            # this assumes we're padding to the same length for all attention
            model_inputs["global_attention_mask"] = len(model_inputs["input_ids"]) * \
                    [[0 for _ in range(len(model_inputs["input_ids"][0]))]]
            # sets all first entries to 1 b/c it's a reference
            model_inputs["global_attention_mask"][0][0] = 1

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_wiki_asp(examples):
        inputs = examples[text_column] # list of sentences
        topics_and_targets = examples[summary_column] # list of (topic, target)s
        all_inputs = []
        all_targets = []
        for inp, ex_topics_and_targets in zip(inputs, topics_and_targets):
            full_inp = " ".join(inp)
            for topic, target in ex_topics_and_targets:
                all_inputs.append(" ".join([topic, full_inp, topic]))
                all_targets.append(target)

        model_inputs = tokenizer(all_inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(all_targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_all_qds(examples):
        # preprocess function used for the non-extracted QDS task
        model_inputs = []
        all_inputs = []     # input documents
        all_questions = []  # questions
        all_targets = []    # target sequences
        passage_ids = []    # metadata
        question_ns = []    # metadata
        response_ids = []   # metadata
        for response_idx, responses in enumerate(examples['responses']):
            for response in responses:
                n_responses = len(response['summaries'])
                all_inputs += [examples[text_column][response_idx]] * n_responses
                all_questions += examples[question_column][response_idx]
                all_targets += response['summaries']
                response_ids += [response['uid']] * n_responses
                passage_id = examples['metadata'][response_idx]['passage_id']
                passage_ids += [passage_id] * n_responses
                if 'question_n' in examples['metadata'][response_idx]:
                    question_ns += [f"{passage_id}-{question_n}" for question_n in examples['metadata'][response_idx]['question_n']]
                else:
                    question_ns += [f"{passage_id}-{i}" for i in range(n_responses)]

        assert len(all_targets) == len(all_inputs), "Different number of targets and inputs!"
        assert len(all_targets) == len(all_questions), "Different number of targets and questions!"
        assert len(all_targets) == len(passage_ids), "Different number of targets and passage ids"
        assert len(all_targets) == len(question_ns), "Different number of targets and question #s"
        assert len(all_targets) == len(response_ids), "Different number of targets and response ids"

        # appending question leads to truncation
        #inputs = [prefix + inp + question for inp, qst in zip(inputs, questions)]
        inputs = [" ".join([prefix, qst, sep_tok, inp, sep_tok, qst]) for inp, qst in zip(all_inputs, all_questions)]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        if isinstance(model, LEDForConditionalGeneration) and True:
            # create global attention masks to attend to questions
            # I'm not sure how to vectorize this in standard python
            global_attns = []
            for model_input, attn_mask in zip(model_inputs.input_ids, model_inputs.attention_mask):
                idxs = [idx for idx, tok in enumerate(model_input) if tok == sep_tok_id]
                assert len(idxs) in [3, 2], f"Expected 2-3 sep tokens, found {len(sep_tok_idxs)}!"
                global_attn = [0] * len(model_input)
                global_attn[1 :idxs[0] - 1] = [1] * (idxs[0] - 2) # don't attend to sos or sep
                if len(idxs) == 3:
                    assert idxs[0] - 2 == idxs[2] - idxs[1] - 1, "Question tokenized to different lengths!"
                    global_attn[idxs[1] + 1: idxs[2]] = [1] * (idxs[2] - idxs[1] - 1)
                assert len(global_attn) == len(model_input), "Found different length for input ID and global mask!"
                assert len(global_attn) == len(attn_mask), "Found different length for input ID and attention mask!"
                global_attns.append(global_attn)
            model_inputs["global_attention_mask"] = global_attns
        elif isinstance(model, LEDForConditionalGeneration): # pad first <s> token
            global_attns = [[0] * len(i) for i in model_inputs]
            for mask in global_attns:
                mask[0] = 1

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(all_targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["full_refs"] = all_targets
        model_inputs["passage_ids"] = passage_ids
        model_inputs["question_ns"] = question_ns
        model_inputs["response_ids"] = response_ids
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # control dataset size (# examples)
        if "qds" in data_args.train_file and data_args.max_train_stories:
            train_dataset = train_dataset.shuffle()
            train_dataset = train_dataset.select(range(data_args.max_train_stories))
            logger.info(f"Adjusting # train stories to {data_args.max_train_stories}")

        if data_args.max_train_samples is not None and (data_args.train_file is None or "qds" not in data_args.train_file):
            if data_args.max_train_samples > len(train_dataset):
                data_args.max_train_samples = len(train_dataset)
                logger.info(f"max_train_samples > dataset size ({len(train_dataset)})")
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
            logger.info(f"Adjusting train dataset size to {data_args.max_train_samples}")

        if data_args.train_file is not None:
            preprocess_fn = preprocess_all_qds
            logger.info(f"Loading training data from {data_args.train_file}")
        else:
            if data_args.dataset_name == "wiki_asp":
                preprocess_fn = preprocess_wiki_asp
            else:
                preprocess_fn = preprocess_function
            logger.info(f"Loading training data for {data_args.dataset_name}")
        train_dataset = train_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing train data..."
        )

        if data_args.max_train_samples is not None:
            if data_args.max_train_samples > len(train_dataset):
                data_args.max_train_samples = len(train_dataset)
                logger.info(f"max_train_samples > dataset size ({len(train_dataset)})")
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
            logger.info(f"Adjusting train dataset size to {data_args.max_train_samples}")
        if isinstance(model, LEDForConditionalGeneration):
            train_dataset.set_format(columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None and (data_args.validation_file is None or "qds" not in data_args.validation_file):
            if data_args.max_eval_samples > len(eval_dataset):
                data_args.max_eval_samples = len(eval_dataset)
                logger.info(f"max_eval_samples > dataset size ({len(eval_dataset)})")
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
            logger.info(f"Adjusting validation dataset size to {data_args.max_eval_samples}")

        if data_args.validation_file is not None:
            preprocess_fn = preprocess_all_qds
            logger.info(f"Loading eval data for {data_args.validation_file}")
        else:
            if data_args.dataset_name == "wiki_asp":
                preprocess_fn = preprocess_wiki_asp
            else:
                preprocess_fn = preprocess_function
            logger.info(f"Loading eval data for {data_args.dataset_name}")
        eval_dataset = eval_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing eval data..."
        )

        if data_args.max_eval_samples is not None:
            if data_args.max_eval_samples > len(eval_dataset):
                data_args.max_eval_samples = len(eval_dataset)
                logger.info(f"max_eval_samples > dataset size ({len(eval_dataset)})")
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
            logger.info(f"Adjusting validation dataset size to {data_args.max_eval_samples}")
        if isinstance(model, LEDForConditionalGeneration):
            eval_dataset.set_format(columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]

        if data_args.test_file is not None:
            preprocess_fn = preprocess_all_qds
            logger.info(f"Loading prediction data from {data_args.test_file}")
        else:
            if data_args.dataset_name == "wiki_asp":
                preprocess_fn = preprocess_wiki_asp
            else:
                preprocess_fn = preprocess_function
            logger.info(f"Loading prediction data for {data_args.dataset_name}")
        predict_dataset = predict_dataset.map(
            preprocess_fn,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing prediction data..."
        )

        if data_args.max_predict_samples is not None:
            if data_args.max_predict_samples > len(predict_dataset):
                data_args.max_predict_samples = len(predict_dataset)
                logger.info(f"max_predict_samples > dataset size ({len(predict_dataset)})")
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
            logger.info(f"Adjusting test/predict dataset size to {data_args.max_predict_samples}")

        if isinstance(model, LEDForConditionalGeneration):
            predict_dataset.set_format(columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metrics
    # TODO(AW): install comet?
    # NOTE(AW): BLEURT is really trained for this setting and the scores are really bad
    #           BLEU seems redundant with ROUGE
    #metric_names = ["bertscore", "bleurt", "bleu", "google_bleu", "meteor", "rouge"]
    def _load_metric(metric_name):
        if metric_name == "meteor":
            return load_metric("utils/metrics/multi_meteor")
        elif metric_name == "rouge":
            return load_metric("utils/metrics/multi_rouge")
        else:
            return load_metric(metric_name)
    metric_names = ["bertscore", "meteor", "rouge"]
    metrics = [_load_metric(metric_name) for metric_name in metric_names]
    logger.info(f"Evaluating with metrics {','.join(metric_names)}")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        all_results = {}
        for metric in metrics:
            if metric.name == "bert_score":
                result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                for k in ['precision', 'recall', 'f1']:
                    all_results[f"{metric.name}_{k}"] = np.mean(result[k]) * 100
                for k in ['hashcode']:
                    all_results[f"{metric.name}_{k}"] = result[k]

            elif metric.name in ["bleu", "google_bleu"]:
                # options: smoothing (default false)
                tokenized_preds = [tokenizer.tokenize(p) for p in decoded_preds]
                tokenized_refs = [[tokenizer.tokenize(r)] for r in decoded_labels]
                result = metric.compute(predictions=tokenized_preds, references=tokenized_refs)
                if metric.name == "bleu":
                    for k in ['bleu', 'precisions']:
                        all_results[f"{metric.name}_{k}"] = np.mean(result[k]) * 100
                    for k in ['brevity_penalty']:
                        all_results[f"{metric.name}_{k}"] = result[k]
                else:
                    for k, v in result.items():
                        all_results[f"{metric.name}_{k}"] = np.mean(v) * 100

            elif metric.name == "rouge":
                result = metric.compute(predictions=decoded_preds,
                                        references=[[l] for l in decoded_labels],
                                        use_stemmer=True)
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                for k, v in result.items():
                    all_results[f"{metric.name}_{k}"] = np.mean(v)

            else: # bleurt, meteor
                result = metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
                for k, v in result.items():
                    all_results[f"{metric.name}_{k}"] = np.mean(v) * 100

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        all_results["gen_len"] = np.mean(prediction_lens)
        #all_results = {k: round(v, 4) for k, v in all_results.items()}
        return all_results


    def compute_multireference_metrics(references, predictions, example_ids, metric_prefix=""):
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        if not isinstance(references[0], str):
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                references = np.where(references != -100, references, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(references, skip_special_tokens=True)
        else:
            decoded_labels = references

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # Sort by example ids
        ex2pred = {}
        ex2refs = defaultdict(list)
        for pred, ref, ex_id in zip(decoded_preds, decoded_labels, example_ids):
            if ex_id in ex2pred:
                assert pred == ex2pred[ex_id], "Different predictions for the same input!"
            else:
                ex2pred[ex_id] = pred
            ex2refs[ex_id].append(ref)
        unique_example_ids = list(set(example_ids))
        grouped_decoded_preds = [ex2pred[ex] for ex in unique_example_ids]
        grouped_decoded_labels = [ex2refs[ex] for ex in unique_example_ids]

        all_results = {}
        for metric in metrics:
            if metric.name == "bert_score":
                result = metric.compute(predictions=grouped_decoded_preds,
                                        references=grouped_decoded_labels,
                                        lang="en")
                for k in ['precision', 'recall', 'f1']:
                    all_results[f"{metric.name}_{k}"] = np.mean(result[k]) * 100
                for k in ['hashcode']:
                    all_results[f"{metric.name}_{k}"] = result[k]

            elif metric.name in ["bleu", "google_bleu"]:
                # options: smoothing (default false)
                tokenized_preds = [tokenizer.tokenize(p) for p in decoded_preds]
                tokenized_refs = [[[tokenizer.tokenize(r)] for r in rr] for rr in decoded_labels]
                result = metric.compute(predictions=tokenized_preds, references=tokenized_refs)
                if metric.name == "bleu":
                    for k in ['bleu', 'precisions']:
                        all_results[f"{metric.name}_{k}"] = np.mean(result[k]) * 100
                    for k in ['brevity_penalty']:
                        all_results[f"{metric.name}_{k}"] = result[k]
                else:
                    for k, v in result.items():
                        all_results[f"{metric.name}_{k}"] = np.mean(v) * 100

            elif metric.name == "rouge":
                result = metric.compute(predictions=grouped_decoded_preds,
                                        references=grouped_decoded_labels,
                                        use_stemmer=True)
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                for k, v in result.items():
                    all_results[f"{metric.name}_{k}"] = np.mean(v)

            else: # bleurt, meteor
                result = metric.compute(predictions=grouped_decoded_preds, references=grouped_decoded_labels)
                for k, v in result.items():
                    all_results[f"{metric.name}_{k}"] = np.mean(v) * 100

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        all_results["gen_len"] = np.mean(prediction_lens)
        if metric_prefix is not None:
            new_results = {}
            for k, v in all_results.items():
                new_results[f'{metric_prefix}_{k}'] = v
            all_results = new_results
        return all_results

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        train_metrics["train_final_loss"] = train_result.training_loss

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        eval_metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )

        if "qds" in data_args.test_file:
            # multireference metrics
            predict_metrics = compute_multireference_metrics(references=predict_dataset['full_refs'],
                                                             predictions=predict_results.predictions,
                                                             example_ids=predict_dataset['question_ns'],
                                                             metric_prefix="predict"
                                                            )
            for k in ['predict_loss']:
                predict_metrics[k] = predict_results.metrics[k]

        else:
            predict_metrics = predict_results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        predict_metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", predict_metrics)
        trainer.save_metrics("predict", predict_metrics)

        # save predictions
        if trainer.is_world_process_zero() and training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip().replace("\n"," ") for pred in predictions]

            if data_args.include_info_in_prediction_file:
                refs = predict_dataset['full_refs']
                inps = [tokenizer.decode(inp[:100], skip_special_tokens=True, clean_up_tokenization_space=True) for inp in predict_dataset['input_ids']]

            output_prediction_file = os.path.join(training_args.output_dir, data_args.prediction_out_file)
            with open(output_prediction_file, "w") as writer:
                if data_args.include_info_in_prediction_file:
                    assert len(predictions) == len(refs), "Different number of predictions and references!"
                    assert len(predictions) == len(inps), "Different number of predictions and inputs!"
                    for idx, (pred, ref, inp) in enumerate(zip(predictions, refs, inps)):
                        out_d = {
                                 'prediction': pred, 'reference': ref, '(partial) input': inp,
                                 'passage_id': predict_dataset['passage_ids'][idx],
                                 'question_n': predict_dataset['question_ns'][idx],
                                 'response_id': predict_dataset['response_ids'][idx],
                                }
                        writer.write(f"{json.dumps(out_d)}\n")
                else:
                    writer.write("\n".join(predictions))
            logging.info(f"Wrote predictions to {output_prediction_file}")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

