from typing import Any, Union

import numpy as np
from pydantic import Field, field_validator
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from groot.vla.data.transform.base import InvertibleModalityTransform, ModalityTransform

T_Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class LanguageTransform(InvertibleModalityTransform):
    """Transform for language modalities.

    Attributes:
        apply_to (list[str]): The keys in the modality to load and transform.
        tokenizer (T_Tokenizer): The tokenizer to use. Can be either PreTrainedTokenizer, PreTrainedTokenizerFast, or path to the HuggingFace tokenizer
    """

    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    tokenizer: T_Tokenizer = Field(..., description="The tokenizer to use.")

    @field_validator("tokenizer")
    def validate_tokenizer(cls, v: T_Tokenizer | str) -> T_Tokenizer:
        if isinstance(v, str):
            return AutoTokenizer.from_pretrained(v)
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Tokenize the data, pad the tokens to the longest sequence, and concatenate the tokens.

        Args:
            data (dict[str, Any]): The complete data dictionary.

        Returns:
            dict[str, Any]: The processed data dictionary with the keys in `apply_to` replaced with the tokenized data.
        """
        for key in self.apply_to:
            data[key] = self.tokenizer(
                data[key], return_tensors="pt", padding=True, truncation=True
            ).input_ids
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Untokenize the data.

        Args:
            data (dict[str, Any]): The processed data dictionary with the keys in `apply_to` replaced with the tokenized data.

        Returns:
            dict[str, Any]: The untokenized data.
        """
        for key in self.apply_to:
            data[key] = self.tokenizer.decode(data[key], skip_special_tokens=True)
        return data


class LanguageRemovePrefix(ModalityTransform):
    apply_to: list[str] = Field(
        ..., description="The keys in the modality to remove the prefix from."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove the prefix from the language.
        Expects:
        - data[key] is a list of strings, shape (T,)
        - OR data[key] is a list of lists of strings, shape (B, T)

        Args:
            data (dict[str, Any]): The processed data dictionary with the keys in `apply_to` replaced with the tokenized data.

        Returns:
            dict[str, Any]: The data with the prefix removed for key in `apply_to`.
        """
        for key in self.apply_to:
            value = data[key]
            # Handle both batched (list of lists) and non-batched (list) language data.
            if isinstance(value[0], np.ndarray):
                # Batched case: list of lists of strings, shape (B, T)
                data[key] = np.array(
                    [[lang.split(": ")[-1] for lang in sublist] for sublist in value]
                )
            else:
                # Non-batched case: list of strings, shape (T,)
                data[key] = np.array([lang.split(": ")[-1] for lang in value])
        return data
