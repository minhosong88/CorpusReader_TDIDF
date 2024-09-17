# CorpusReader_TFIDF

**CorpusReader_TFIDF** is a custom Python class designed to calculate **TF-IDF** (Term Frequency-Inverse Document Frequency) for documents in a corpus. This class leverages the NLTK library and allows for flexible options such as stopword removal, stemming, and case normalization. It was developed as part of an assignment for the **Introduction to Natural Language Processing (NLP)** class.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Example](#example)
- [Methods](#methods)
  - [tf_idf](#tf_idf)
  - [tf_idf_all](#tf_idf_all)
  - [cosine_sim](#cosine_sim)
  - [query](#query)
- [Customization Options](#customization-options)
- [License](#license)

## Features

- **TF-IDF Calculation**: Computes TF-IDF values for terms within a document or across a corpus.
- **Stopword Handling**: Option to remove standard NLTK stopwords or supply a custom stopword list.
- **Stemming**: Supports stemming using NLTK's Snowball stemmer.
- **Cosine Similarity**: Calculates the cosine similarity between documents or between a new document and existing documents.
- **Preprocessing Options**: Offers options for case normalization, stemming order, and custom term processing.
- **Designed for Learning**: Developed as part of an NLP class assignment, it’s designed to illustrate key NLP concepts like term weighting and document similarity.

## Installation

To use **CorpusReader_TFIDF**, first ensure you have the required dependencies:

1. **Install NLTK**:

```bash
 pip install nltk
```

2. **Install NumPy:**

```bash
pip install numpy
```

3. **Download NLTK Corpora:** If you plan to use corpora from NLTK (such as genesis), download them using:

```bash
import nltk
nltk.download('genesis')
```

## Usage

### Example

```python
from nltk.corpus import genesis
from corpusreader_tfidf import CorpusReader_TFIDF

# Initialize the CorpusReader_TFIDF with the genesis corpus
tfidf_reader = CorpusReader_TFIDF(genesis, tf="log", idf="smooth", stopword="standard", to_stem=True)

# Calculate TF-IDF for a specific document
print(tfidf_reader.tf_idf('english-kjv.txt'))

# Calculate TF-IDF for all documents in the corpus
print(tfidf_reader.tf_idf_all())

# Query for documents similar to a new set of words
new_doc = ['the', 'lord', 'commanded']
print(tfidf_reader.query(new_doc))
```

## Methods

### `tf_idf(fileid, return_zero=False)`

Calculates the TF-IDF values for a specific document in the corpus.

- **Arguments**:

  - `fileid` (str): The document ID.
  - `return_zero` (bool): Whether to include terms with zero TF-IDF scores.

- **Returns**: A dictionary where the keys are terms and the values are the corresponding TF-IDF scores.

### `tf_idf_all(return_zero=False)`

Calculates the TF-IDF values for all documents in the corpus.

- **Arguments**:

  - `return_zero` (bool): Whether to include terms with zero TF-IDF scores.

- **Returns**: A dictionary where the keys are document IDs and the values are dictionaries of TF-IDF scores.

### `cosine_sim(fileid_1, fileid_2)`

Calculates the cosine similarity between two documents in the corpus.

- **Arguments**:

  - `fileid_1` (str): The ID of the first document.
  - `fileid_2` (str): The ID of the second document.

- **Returns**: A float representing the cosine similarity score.

### `query(words: List[str])`

Finds documents in the corpus that are most similar to a new document based on cosine similarity.

- **Arguments**:

  - `words` (List[str]): A list of words representing the new document.

- **Returns**: A list of tuples, where each tuple contains a document ID and its cosine similarity score, sorted in descending order of similarity.

## Customization Options

When initializing the `CorpusReader_TFIDF` class, several options are available:

- **tf**: Choose between `"raw"` (raw term frequency) and `"log"` (logarithmic term frequency).
- **idf**: Choose between `"base"` (standard inverse document frequency) and `"smooth"` (smoothed inverse document frequency).
- **stopword**: Choose between `"none"`, `"standard"` (NLTK stopwords), or provide a custom stopword file.
- **to_stem**: Enable stemming using the Snowball stemmer (`True` or `False`).
- **stem_first**: If stemming is enabled, choose whether to stem before removing stopwords (`True` or `False`).
- **ignore_case**: Whether to normalize words to lowercase (`True` or `False`).

## License

This project is part of the **Introduction to Natural Language Processing (NLP)** course assignment and is intended for educational purposes.
