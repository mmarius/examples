# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Face BERT wrapped inside a :class:`.ComposerModel`."""

from __future__ import annotations

from typing import Optional, Mapping

from composer.metrics.nlp import (BinaryF1Score, LanguageCrossEntropy,
                                  MaskedAccuracy, InContextLearningLMAccuracy)
from composer.models.huggingface import HuggingFaceModel
from composer.utils.import_helpers import MissingConditionalImportError
from transformers import GPTNeoXForCausalLM, GPTNeoXLayer

import os

__all__ = ['create_hf_pythia_lm',]


def create_hf_pythia_lm(
        num_labels: int,
        pretrained_model_name: str = 'EleutherAI/pythia-70m',
        revision: str = 'main',
        use_pretrained: Optional[bool] = False,
        model_config: Optional[dict] = None,
        tokenizer_name: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = False):
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp',
                                            conda_package='transformers') from e

    if not model_config:
        model_config = {}

    model_config['num_labels'] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = 'EleutherAI/pythia-70m'

    # create cache dir for revisions
    cache_dir = os.environ['TRANSFORMERS_CACHE']
    cache_dir = os.path.join(cache_dir, "pythia", revision)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)  

    if use_pretrained:
        # assert transformers.AutoModelForCausalLM.from_pretrained is not None, 'AutoModelForCausalLM has from_pretrained method'
        model = GPTNeoXForCausalLM_Wrapper.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name, revision=revision, cache_dir=cache_dir, **model_config)
    else:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name, **model_config)
        # assert transformers.AutoModelForCausalLM.from_config is not None, 'AutoModelForCausalLM has from_config method'
        model = GPTNeoXForCausalLM_Wrapper.from_config(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

    metrics = [
        # LanguageCrossEntropy(ignore_index=-100), # TODO(mm): this is not aware of shifted labels
        InContextLearningLMAccuracy()
    ]
    return HuggingFaceModel(model=model,
                            tokenizer=tokenizer,
                            use_logits=True,
                            metrics=metrics,
                            shift_labels=True)


# # We overwrite the composer HuggingFaceModel class for debugging purposes
# class MyHuggingFaceModel(HuggingFaceModel):
#     def forward(self, batch):        
#         if isinstance(batch, Mapping):
#             # Further input validation is left to the huggingface forward call
#             batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
#             # print(batch["input_ids"][0])
#             # print(batch["labels"][0])
#             output = self.model(**batch)  # type: ignore (thirdparty)
#             # print(output.loss)
#             # assert False
#         else:
#             raise ValueError(
#                 'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
#             )
#         return output

class GPTNeoXForCausalLM_Wrapper(GPTNeoXForCausalLM):
    # We overwrite the GPTNeoXForCausalLM class to add support for distributed training
    
    # FSDP Wrap Function
    def fsdp_wrap_fn(self, module):
        return isinstance(module, GPTNeoXLayer) # only GPTNeoXLayer modules will be split across GPUs

    # Activation Checkpointing Function
    def activation_checkpointing_fn(self, module):
        return isinstance(module, GPTNeoXLayer) # only GPTNeoXLayer modules will be split across GPUs