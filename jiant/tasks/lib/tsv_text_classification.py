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
from jiant.tasks.lib.templates.shared import single_sentence_featurize, double_sentence_featurize, labels_to_bimap

@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: str
    label_id: int

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label_id=self.label_id,
        )


@dataclass
class ExampleTwo(BaseExample):
    guid: str
    text_a: str
    text_b: str
    label: str
    label_id: int

    def tokenize(self, tokenizer):
        return TokenizedExampleTwo(
            guid=self.guid,
            text_a=tokenizer.tokenize(self.text_a),
            text_b=tokenizer.tokenize(self.text_b),
            label_id=self.label_id,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class TokenizedExampleTwo(BaseTokenizedExample):
    guid: str
    text_a: List
    text_b: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.text_a,
            input_tokens_b=self.text_b,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class TSVTextClassificationTask(Task):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION

    def __init__(self, name: str, path_dict: dict, **kwargs):
        super().__init__(name, path_dict)

        self.LABELS = [line.strip() for line in open(kwargs["labels_path"])]
        self.LABEL_TO_ID, self.ID_TO_LABEL = labels_to_bimap(self.LABELS)

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
                label = row[1].strip()
            except IndexError: #test
                label=self.LABELS[0]
                
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text=text,
                    label=label,
                    label_id=self.LABEL_TO_ID[label]
                )
            )
        return examples


class TSVTwoTextClassificationTask(Task):
    Example = ExampleTwo
    TokenizedExample = TokenizedExampleTwo
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION

    def __init__(self, name: str, path_dict: dict, **kwargs):
        super().__init__(name, path_dict)

        self.LABELS = [line.strip() for line in open(kwargs["labels_path"])]
        self.LABEL_TO_ID, self.ID_TO_LABEL = labels_to_bimap(self.LABELS)

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
                label = row[2].strip()
            except IndexError:  # test
                label = self.LABELS[0]
            
            examples.append(
                ExampleTwo(
                    guid="%s-%s" % (set_type, i),
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    label_id=self.LABEL_TO_ID[label]
                )
            )
        return examples

