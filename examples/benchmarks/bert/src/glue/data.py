# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np

from composer.utils import MissingConditionalImportError, dist
from omegaconf import DictConfig

_task_column_names = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
}

log = logging.getLogger(__name__)


def create_glue_dataset(
    task: str,
    tokenizer_name: str,
    split: str,
    dataset_cfg: DictConfig,
    num_samples: int = -1,
    max_seq_length: int = 256,
    max_retries: int = 10,
    num_workers: int = 0,
    ft_type: str = "pbft",
):
    try:
        import datasets
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    if task not in _task_column_names:
        raise ValueError(
            f'task ({task}) must be one of {_task_column_names.keys()}')

    if (max_seq_length % 8) != 0:
        log.warning(
            'For performance, a max_seq_length as a multiple of 8 is recommended.'
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name)  #type: ignore (thirdparty)

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        'glue',
        task,
        split=split,
        download_config=download_config,
    )

    # (Optional) select a random subset of the dataset
    if num_samples > 0 and num_samples <= len(dataset):
        print(
        f'Selecting a random subset of size {num_samples}')
        indices = np.random.choice(np.arange(0, len(dataset)), size=num_samples, replace=False)
        dataset = dataset.select(indices)

    log.info(
        f'Starting tokenization by preprocessing over {num_workers} threads!')
    text_column_names = _task_column_names[task]

    if ft_type == "ft":
    
        # ------------ tokenize function for clasifier-based finetuning ------------

        def tokenize(inp):
            # truncates sentences to max_length or pads them to max_length

            first_half = inp[text_column_names[0]]
            second_half = inp[
                text_column_names[1]] if text_column_names[1] in inp else None
            return tokenizer(
                text=first_half,
                text_pair=second_half,
                padding='max_length',
                max_length=max_seq_length,
                truncation=True,
            )
        
        # --------------------------------------------------------------------------

        num_labels = len(dataset.features["label"].names)
        tokenize_function = tokenize
        label_column = "label"

    elif ft_type == "pbft":
    
        # ------------ tokenize function for pattern-based finetuning ------------
        
        # construct verbalizer
        mappings = dataset_cfg.verbalizer.split(",")
        mappings = [map.split("-->") for map in mappings]
        verbalizer = {token.strip(): int(label) for token, label in mappings}
        verbalizer_inv = {v: k for k, v in verbalizer.items()} # invert

        # we need to update the number of classes of the dataset
        new_features = dataset.features.copy()
        new_features["label"] = datasets.ClassLabel(names=list(tokenizer.vocab.values()))
        dataset = dataset.cast(new_features)
        num_labels = len(dataset.features["label"].names)
        
        def tokenize_with_pattern(inp):
            # apply pattern to inputs
            pattern_examples = [
                dataset_cfg.pattern.format(
                    text1=inp[text_column_names[0]][idx],
                    text2=inp[text_column_names[1]][idx] if text_column_names[1] is not None else None,
                    mask=tokenizer.mask_token)
                for idx in range(len(inp[text_column_names[0]]))
            ]
            
            # tokenizer
            args = (pattern_examples,)
            result = tokenizer(
                *args, 
                padding='max_length',
                max_length=max_seq_length, 
                truncation=True
                )
            
            # # get tokens
            # result["input_tokens"] = [tokenizer.convert_ids_to_tokens(
            #     ids) for ids in result["input_ids"]]

            # # decode input
            # result["input_text"] = [tokenizer.decode(
            #     ids) for ids in result["input_ids"]]

            # convert the dataset label to a token id using the verbalizer
            label_tokens = [verbalizer_inv[label] for label in inp["label"]]
            label_ids = tokenizer.convert_tokens_to_ids(label_tokens)
            
            # place the actual label at the index of the [MASK] token
            result["labels"] = []
            for idx in range(len(result["input_ids"])):
                # TODO(mm): handle the case where the [MAKS] token got removed due to truncation
                try:
                    mask_position = result["input_ids"][idx].index(tokenizer.mask_token_id) # get mask index
                except ValueError: # input got truncated
                    mask_position = -1 # replace the last token with the mask token
                    result["input_ids"][idx][-1] = tokenizer.mask_token_id

                labels = [-100] * max_seq_length # cross entropy will ignore all token positions with label -100
                labels[mask_position] = label_ids[idx] # replace placeholder label at mask position
                result["labels"].append(labels)
        
            return result
        
        # --------------------------------------------------------------------------

        tokenize_function = tokenize_with_pattern
        label_column = "labels"

    elif ft_type == "causal-pbft":
    
        # ------------ tokenize function for pattern-based finetuning with causal LMs ------------
        
        # construct verbalizer
        mappings = dataset_cfg.verbalizer.split(",")
        mappings = [map.split("-->") for map in mappings]
        verbalizer = {token.strip(): int(label) for token, label in mappings}
        verbalizer_inv = {v: k for k, v in verbalizer.items()} # invert

        # we need to update the number of classes of the dataset
        new_features = dataset.features.copy()
        new_features["label"] = datasets.ClassLabel(names=list(tokenizer.vocab.values()))
        dataset = dataset.cast(new_features)
        num_labels = len(dataset.features["label"].names)

        # add padding token to tokenizer (pythia doesn't have a padding token)
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_with_pattern(inp):
            # apply pattern to inputs
            pattern_examples = [
                dataset_cfg.pattern.format(
                    text1=inp[text_column_names[0]][idx],
                    text2=inp[text_column_names[1]][idx] if text_column_names[1] is not None else None,
                    label=verbalizer_inv[inp["label"][idx]][1:] # we omit the Ġ
                )
                for idx in range(len(inp[text_column_names[0]]))
            ]
            
            # tokenizer
            args = (pattern_examples,)
            result = tokenizer(
                *args, 
                padding='max_length',
                max_length=max_seq_length, 
                truncation=True
                )
            
            # # DEBUGGING ONLY 
            # # get tokens
            # result["input_tokens"] = [tokenizer.convert_ids_to_tokens(
            #     ids) for ids in result["input_ids"]]

            # # decode input
            # result["input_text"] = [tokenizer.decode(
            #     ids) for ids in result["input_ids"]]

            # convert the dataset label to a token id using the verbalizer
            label_tokens = [verbalizer_inv[label] for label in inp["label"]]
            label_ids = tokenizer.convert_tokens_to_ids(label_tokens)
            
            # place the actual label at the end of the pattern
            # the input looks like this: pattern [PAD] ... [PAD]
            result["labels"] = []
            result["continuation_indices"] = []
            for idx in range(len(result["input_ids"])):
                # get the position of the first [PAD] token
                try:
                    pad_position = result["input_ids"][idx].index(tokenizer.pad_token_id) # get index of first padding token
                    label_position = pad_position - 1 # we place the label before the padding token (the target token is always the last token before padding)
                except ValueError: # input got truncated
                    label_position = len(result["input_ids"][idx]) - 1 # use the last token

                labels = [-100] * max_seq_length # cross entropy will ignore all token positions with label -100
                labels[label_position] = label_ids[idx] # replace placeholder label
                result["labels"].append(labels)
                result["continuation_indices"].append(label_position)

            return result
        
        # --------------------------------------------------------------------------------------

        tokenize_function = tokenize_with_pattern
        label_column = "labels"

    else:
        raise ValueError(f'Unsupported ft_type={ft_type}')

    columns_to_remove = ['idx'
                        ] + [i for i in text_column_names if i is not None]

    assert isinstance(dataset, datasets.Dataset)
    safe_name = tokenizer_name.replace('/', ',')

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        new_fingerprint=f'{task}-{safe_name}-tokenization-{split}',
        load_from_cache_file=False,
    )

    # format dataset
    columns = ["input_ids", "token_type_ids", "attention_mask", label_column]
    if "roberta" in str(type(tokenizer)):
        columns.remove("token_type_ids")
    if "gpt_neox" in str(type(tokenizer)): # pythia's tokenizer type
        columns.remove("token_type_ids")
        columns.append("continuation_indices")

    dataset.set_format(type="torch", columns=columns, output_all_columns=False)

    return dataset, num_labels