
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only runs once)
nltk.download('punkt',      quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)

# Initialize lemmatizer (converts words to root form: "running" → "run")
lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Remove extra whitespace
    - Remove special characters (keep letters, digits, punctuation)
    - Strip leading/trailing spaces
    """
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-ASCII characters (emojis, weird symbols)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text.strip()


def tokenize_sentences(text: str) -> list[str]:
    """
    Split a paragraph into individual sentences.
    Example: "Hello world. How are you?" → ["Hello world.", "How are you?"]
    """
    return sent_tokenize(text)


def tokenize_words(text: str) -> list[str]:
    """
    Split text into individual words (tokens).
    Example: "Hello world" → ["Hello", "world"]
    """
    return word_tokenize(text)


def remove_stopwords(words: list[str]) -> list[str]:
    """
    Remove common words that don't add meaning (the, is, at, which...).
    Only keeps meaningful content words.
    """
    return [w for w in words if w.lower() not in STOP_WORDS and w.isalpha()]


def lemmatize_words(words: list[str]) -> list[str]:
    """
    Convert words to their base/root form.
    Examples: "running" → "run", "better" → "good", "cats" → "cat"
    """
    return [lemmatizer.lemmatize(w.lower()) for w in words]


def preprocess_text(text: str) -> dict:
    """
    Full preprocessing pipeline — runs all steps in order.

    Returns a dictionary with:
    - cleaned_text     : clean version of original text
    - sentences        : list of sentences
    - tokens           : all words
    - filtered_tokens  : words without stopwords
    - lemmatized_tokens: root-form words
    """
    # Clean
    cleaned = clean_text(text)

    # Split into sentences
    sentences = tokenize_sentences(cleaned)

    # Split into words
    tokens = tokenize_words(cleaned)

    # Remove stopwords
    filtered = remove_stopwords(tokens)

    # Lemmatize
    lemmatized = lemmatize_words(filtered)

    return {
        "cleaned_text":      cleaned,
        "sentences":         sentences,
        "tokens":            tokens,
        "filtered_tokens":   filtered,
        "lemmatized_tokens": lemmatized,
    }



