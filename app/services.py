import os
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class EmbeddingModelLoader:
    _instance = None

    # this will make sure even if many objects are created, the model is loaded only once
    def __new__(cls):
        if cls._instance is None:
            print("Loading SentenceTransformer ('all-MiniLM-L6-v2) into memory")
            cls._instance = super().__new__(cls)
            # load the cache the model
            cls._instance.model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    @model.setter
    def model(self, transformer_model: SentenceTransformer):
        self._model = transformer_model


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """Generate the embeddings of the chunks passed, created 384 dimensional dense vector"""

    if not chunks:
        return []

    loader = EmbeddingModelLoader()
    embeddings = loader.model.encode(chunks, convert_to_numpy=True)

    return embeddings.tolist()


def extract_text(file_path: str):
    """Extracts raw text from .pdf or .txt file and returns a raw string"""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()

    # to handle .txt files
    if extension == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as file:
                return file.read()

    # now finally to handle the .pdf files
    elif extension == ".pdf":
        try:
            reader = PdfReader(file_path)
            extracted_pages = []

            for index, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""

                if index > 0:
                    extracted_pages.append(f"\n--- Page {index + 1} ---\n")
                extracted_pages.append(page_text)

            return "".join(extracted_pages)

        except Exception as e:
            raise ValueError(
                f"failed to extract text from pdf: {file_path}, Error: {str(e)}"
            )

    # for unsuported file type
    else:
        raise ValueError(
            f"Unsupported file extension '{extension}'. Only pdf and txt files are supported."
        )


def chunk_test_fixed_size(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    """Split text into fixed size chunks with overlaping and return a list of the chunks"""

    if chunk_size <= 0:
        raise ValueError("chunksize must be positve integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller then chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    # checking for base condition
    if (
        text_length <= chunk_size
    ):  # the total text in the file is less then the chunks size
        if text_length > 0:  # and that it is not emtpy
            chunks.append(text)  # just crate a chunk of the entire text
        return chunks

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk:
            chunks.append(chunk)

        # Move forward
        start += chunk_size - overlap

        if (start >= text_length) or (end >= text_length):
            break

    return chunks


def chunk_text_sentence(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """Splits text on sentence boundary and groups a fixed number of sentence together to chunk them"""

    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive integer")

    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append(" ".join(group))

    return chunks


def chunk_text(text: str, strategy: str, **kwargs) -> list[str]:
    """router function to route to the choosen chunking strategy"""
    if strategy == "fixed_size":
        return chunk_test_fixed_size(text, **kwargs)
    elif strategy == "semantic":
        return chunk_text_sentence(text, **kwargs)
    else:
        raise ValueError(
            f"Unkown chunking strategy '{strategy}'. Use 'fixed_size' or 'semantic' chunking strategy."
        )
