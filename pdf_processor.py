import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English


# PDF reading function
def text_formatter(text: str) -> str:
    """Format the extracted PDF text by removing line breaks and stripping spaces."""
    return text.replace("\n", " ").strip()


def read_pdf(pdf_file) -> list[dict]:
    """Reads a PDF file and extracts text, word, and sentence counts per page."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages_and_text = []
    for page_number, page in tqdm(enumerate(doc), desc="Reading PDF"):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_text.append(
            {
                "page_number": page_number + 1,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text)
                / 4,  # Rough estimate: 1 token ~ 4 characters
                "text": text,
            }
        )
    return pages_and_text


def process_chunks(pages_and_text: list[dict]) -> list[dict]:
    """Processes each page of text into chunks of sentences, returns chunk data."""
    nlp = English()
    nlp.add_pipe("sentencizer")

    for item in tqdm(pages_and_text, desc="Splitting Sentences"):
        item["sentences"] = [str(sent) for sent in nlp(item["text"]).sents]
        item["sentence_chunks"] = split_list(item["sentences"], 10)
        item["num_of_chunks"] = len(item["sentence_chunks"])

    pages_and_chunks = []
    for item in tqdm(pages_and_text, desc="Creating Chunks"):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": " ".join(sentence_chunk).strip(),
                "chunk_word_count": sum(
                    len(sentence.split()) for sentence in sentence_chunk
                ),  # Accurate word count
                "chunk_token_count": sum(len(sentence) for sentence in sentence_chunk)
                / 4,  # Refined token estimation
            }
            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks


# Helper function for splitting sentences into chunks
def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
    """Splits a list into smaller sublists of a specified slice size."""
    return [
        input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)
    ]
