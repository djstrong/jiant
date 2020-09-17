import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import construct_double_input_tokens_and_segment_ids, \
    create_input_set_from_tokens_and_segments, \
    construct_single_input_tokens_and_segment_ids, labels_to_bimap


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: float

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label=self.label,
        )


@dataclass
class ExampleTwo(BaseExample):
    guid: str
    text_a: str
    text_b: str
    label: float

    def tokenize(self, tokenizer):
        return TokenizedExampleTwo(
            guid=self.guid,
            text_a=tokenizer.tokenize(self.text_a),
            text_b=tokenizer.tokenize(self.text_b),
            label=self.label,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label: float

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.text,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            label=self.label,
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class TokenizedExampleTwo(BaseTokenizedExample):
    guid: str
    text_a: List
    text_b: List
    label: float

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_double_input_tokens_and_segment_ids(
            input_tokens_a=self.text_a,
            input_tokens_b=self.text_b,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            label=self.label,
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label: float
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label: torch.FloatTensor
    tokens: list


class TSVTextRegressionTask(Task):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.REGRESSION

    def __init__(self, name: str, path_dict: dict, **kwargs):
        super().__init__(name, path_dict)

        self.evaluation_scheme=kwargs["evaluation_scheme"]

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    def _create_examples(self, path, set_type):
        examples = []

        for i, line in enumerate(open(path)):
            row = line.split('\t')
            text = row[0].strip()
            try:
                label = float(row[1].strip())
            except IndexError:  # test
                label = 0.0
            
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text=text,
                    label=label
                )
            )
        return examples


class TSVTwoTextRegressionTask(Task):
    Example = ExampleTwo
    TokenizedExample = TokenizedExampleTwo
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.REGRESSION

    def __init__(self, name: str, path_dict: dict, **kwargs):
        super().__init__(name, path_dict)

        self.evaluation_scheme=kwargs["evaluation_scheme"]

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    def _create_examples(self, path, set_type):
        examples = []

        for i, line in enumerate(open(path)):
            row = line.split('\t')
            text_a = row[0].strip()
            text_b = row[1].strip()
            try:
                label = float(row[2].strip())
            except IndexError:  # test
                label = 0.0
            
            examples.append(
                ExampleTwo(
                    guid="%s-%s" % (set_type, i),
                    text_a=text_a,
                    text_b=text_b,
                    label=label
                )
            )
        return examples
