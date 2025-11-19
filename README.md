# Twitch & Discord Chat Topic Modeling

## Project Overview

This project conducts topic modeling analysis on user chat data from two social media platforms: Twitch and Discord. The core objective is to explore and compare user discussion themes across different platforms and using different algorithms.

The project utilizes a 2x2 experimental design for comparative analysis by controlling key variables:

  * **Platform:** Twitch vs. Discord
  * **Algorithm:** LDA (Latent Dirichlet Allocation) vs. GSDMM (Gibbs Sampling Dirichlet Multinomial Mixture)

## Methodology: How Topic Modeling Was Done

The analysis is executed via a series of Jupyter Notebook scripts. The core workflow for each script is as follows:

### 1\. Data Loading

  * The script first mounts Google Drive to access the datasets stored in the cloud.
  * It iterates through all `.csv` files in the specified platform's data directory (e.g., `/content/drive/My Drive/Twitch dataset/Cleaned data/`).
  * Chat messages are loaded from the `Message` (Twitch) or `Content` (Discord) column into a pandas DataFrame.

### 2\. Standardized Preprocessing Pipeline

To strictly control experimental variables, all scripts use a standardized text preprocessing pipeline:

1.  **Text Normalization:** All text is converted to lowercase, and all non-alphanumeric characters (e.g., punctuation, symbols) are removed.
2.  **Tokenization:** The cleaned text is split into individual words (Tokens).
3.  **Named Entity Recognition (NER):** Using spaCy's `en_core_web_sm` model, named entities (like usernames, locations, organizations) are identified and filtered out to prevent them from being incorrectly classified as topics.
4.  **Part-of-Speech (POS) Tagging:** NLTK is used to assign a part of speech (e.g., noun, verb, adjective) to each token.
5.  **POS Filtering:** **A key control step in this project: only tokens identified as nouns (POS tags starting with `NN`) are retained.** This focuses the models exclusively on the "things" users are discussing.
6.  **Lemmatization:** All remaining nouns are reduced to their base form (e.g., "messages" becomes "message") using NLTK's `WordNetLemmatizer`.
7.  **Stopword Removal:** A customized, multi-layered list of stopwords is used to filter out noise (see "Variable Control" for details).
8.  **Length Filtering:** Any remaining tokens with fewer than 3 characters are removed.

### 3\. Topic Modeling Algorithms

After preprocessing, the resulting list of nouns for each message is fed into one of the following two algorithms:

#### A) LDA (Latent Dirichlet Allocation)

  * **Scripts:**
      * `V3 of Topic Modeling Twitch LDA.ipynb`
      * `V3 of Topic Modeling Discord LDA.ipynb`
  * **Summary:** LDA is the industry-standard topic model, used here as a baseline.
  * **Process:**
    1.  A `gensim` Dictionary and Bag-of-Words (Corpus) are created.
    2.  **K-Value Optimization:** To find the optimal number of topics, the script iterates through different K values (from 2 to 20).
    3.  An LDA model is trained for each K value and evaluated using `c_v` (coherence value).
    4.  The model with the **highest coherence score** is selected as the "best model" for that dataset.

#### B) GSDMM (Gibbs Sampling Dirichlet Multinomial Mixture)

  * **Scripts:**
      * `V3 of Topic Modeling Twitch GSDMM.ipynb`
      * `V3 of Topic Modeling Discord GSDMM.ipynb`
  * **Summary:** GSDMM is a topic modeling algorithm specifically designed for short texts (like chat messages or tweets).
  * **Process:**
    1.  The `MovieGroupProcess` model is initialized.
    2.  In this project, the number of topics for GSDMM is **fixed at `K=20`** with `n_iters=30` (30 iterations) to cluster the documents.

### 4\. Visualization and Output

  * For each processed `.csv` file, the script generates a PDF report visualizing the top 5 topics.
  * The report includes:
      * **Word Clouds:** A visual representation of high-frequency words in each topic.
      * **Bar Charts:** A plot of the top 10 words and their weights.
      * **Network Graphs:** A graph showing the co-occurrence relationships between the top 10 words.

-----

## Experimental Variable Control

To ensure a fair comparison between platforms (Twitch vs. Discord) and algorithms (LDA vs. GSDMM), variables were strictly controlled in the following ways:

1.  **Standardized Preprocessing Pipeline:**

      * All "V3" series scripts use the **exact same** `preprocess` function.
      * This ensures that all text, regardless of platform or algorithm, was cleaned, filtered (**especially the noun-only filter**), and lemmatized identically.

2.  **Systematic Stopword Lists:**

      * This is the **most critical variable control** in the project.
      * All "V3" scripts share a large base stopword list (`general_stopwords`) and domain-specific lists (`aigc_stopwords`, `contextual_stopwords`).
      * The only **deliberate and controlled adjustment** is the platform-specific stopword list:
          * When analyzing Twitch, `twitch_stopwords` is added (removing 'stream', 'chat', 'vod', etc.).
          * When analyzing Discord, `discord_stopwords` is added (removing 'server', 'discord', etc.).
      * This control ensures the models do not simply "discover" obvious, platform-inherent words (like "stream") as major topics, allowing for a deeper and more meaningful comparison of *actual* user conversation themes.

3.  **Consistent Model Parameters (or Selection Method):**

      * **For GSDMM:** Key parameters were held constant for both platforms (`K=20`, `n_iters=30`).
      * **For LDA:** Instead of fixing K, the scripts use a **consistent evaluation method** (`evaluate_coherence` function) to automatically find the optimal K for each dataset. This controls the *process* of model selection, ensuring fairness.

4.  **Isolated Data Sources:**

      * The scripts are functionally identical but point to two different data folders (`Twitch dataset` vs. `Discord dataset`). This isolates the **platform** as the primary independent variable being tested.

## Experiment Version Comparison: "V3" vs. "Copy of..."

The files you provided contain two series of experiments: "V3" scripts and "Copy of V3" scripts. They differ critically in how they **control preprocessing variables**, which directly impacts the topic modeling results.

### 1\. "Copy of..." Scripts (Baseline Control)

In the "Copy of..." series (e.g., `Copy of V3 of Topic Modeling Twitch LDA.ipynb`), the preprocessing is more basic:

  * **Variable Control:** They use a **general stopword list** (`general_stopwords`).
  * **Effect:** This method removes common English words (like "the", "is") and some high-frequency chat words (like "would", "really").
  * **Limitation:** It **does not** control for domain-specific or platform-specific terms. Therefore, the models are likely to extract obvious words like "twitch," "stream," "discord," "ai," or "model" as major "topics." This is noise, not a valuable insight.

### 2\. "V3" Scripts (Strict Control)

In the "V3" series (e.g., `V3 of Topic Modeling Twitch GSDMM.ipynb`), the preprocessing is far more rigorous:

  * **Variable Control:** They use a **multi-layered, custom stopword list**.
  * **Effect:** This is a more advanced form of variable control. By **intentionally adding** the following lists, the scripts precisely "mask" known contextual noise:
      * `aigc_stopwords`: Removes AI-related terms (e.g., 'ai', 'model', 'image').
      * `contextual_stopwords`: Removes query-related terms (e.g., 'question', 'help').
      * `twitch_stopwords` / `discord_stopwords`: The **most critical control**, removing platform-specific terms (e.g., 'stream', 'chat', 'server').
  * **Conclusion:** This method forces the models to find deeper, user-driven conversation themes that exist *beyond* the obvious context of "we are on Twitch talking about AI." It makes the comparison between Twitch and Discord far more meaningful, as the results are not contaminated by self-evident topics.

-----

## How to Run the Project

1.  **Prerequisites:**

      * A Google Account with access to Google Colab and Google Drive.
      * The cleaned `.csv` chat datasets.

2.  **Setup:**

      * Upload your datasets to Google Drive in the expected paths:
          * `/content/drive/My Drive/Twitch dataset/Cleaned data/`
          * `/content/drive/My Drive/Discord dataset/Cleaned data/`
      * Open one of the four main `.ipynb` files in Google Colab.

3.  **Install Dependencies:**

      * Run the first code cell to install all required Python packages:
        ```bash
        !pip install git+https://github.com/rwalk/gsdmm.git
        !pip install nltk spacy gensim pyLDAvis networkx wordcloud
        ```
      * The script will also automatically download NLTK resources (`stopwords`, `wordnet`) and load the spaCy model (`en_core_web_sm`).

4.  **Execution:**

      * In Colab, run all cells (e.g., "Runtime" \> "Run all").
      * The script will:
        1.  Prompt you to mount your Google Drive.
        2.  Process each `.csv` file in the specified folder one by one.
        3.  Print sample messages from the discovered topics to the console.
        4.  Generate a PDF report with visualizations and trigger a download prompt for it in your browser.
