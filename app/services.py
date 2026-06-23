import os
from pypdf import PdfReader


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
