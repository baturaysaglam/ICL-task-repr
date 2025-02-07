## Datasets

This directory has two main subdirectories with task datasets, all in `.json` format. The datasets were originally put together and published by [Todd et al.](https://github.com/ericwtodd/function_vectors). We haven't evaluated our method on some of them since our main focus is functional regression. For more details on the datasets and preprocessing, check out the authors' [paper](https://arxiv.org/abs/2310.15213).

- **`abstractive/`** – Includes tasks where answering requires info that’s not directly in the prompt.
- **`extractive/`** – Includes tasks where the answer is already in the prompt, and the model just needs to find it.

The **`generate/`** directory has scripts we used to filter existing datasets, a notebook for creating new ones, and tools for cleaning and filtering extra datasets.
