import subprocess
import os
import shutil
import string
import random
from pypdf import PdfReader
import ocrmypdf
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")
model.to(device="cuda")


def embed(queries, chunks) -> dict[str, list[tuple[str, float]]]:
    query_embeddings = model.encode(queries, prompt_name="query")
    document_embeddings = model.encode(chunks)

    scores = query_embeddings @ document_embeddings.T
    results = {}
    for query, query_scores in zip(queries, scores):
        chunk_idxs = [i for i in range(len(chunks))]
        # Get a structure like {query: [(chunk_idx, score), (chunk_idx, score), ...]}
        results[query] = list(zip(chunk_idxs, query_scores))

    return results


def random_word(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def convert_pdf(input_file) -> str:
    reader = PdfReader(input_file)
    text = extract_text_from_pdf(reader)

    # Check if there are any images
    image_count = 0
    for page in reader.pages:
        image_count += len(page.images)

    # If there are images and not much content, perform OCR on the document
    if image_count > 0 and len(text) < 1000:
        out_pdf_file = input_file.replace(".pdf", "_ocr.pdf")
        ocrmypdf.ocr(input_file, out_pdf_file, force_ocr=True)

        # Re-extract text
        text = extract_text_from_pdf(PdfReader(input_file))

        # Delete the OCR file
        os.remove(out_pdf_file)

    return text


def extract_text_from_pdf(reader):
    full_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if len(text) > 0:
            full_text += f"---- Page {idx} ----\n" + page.extract_text() + "\n\n"

    return full_text.strip()


def convert_pandoc(input_file, filename) -> str:
    # Temporarily copy the file
    shutil.copyfile(input_file, filename)

    # Convert the file to markdown with pandoc
    output_file = f"{random_word(16)}.md"
    result = subprocess.call(["pandoc", filename, "-t", "markdown", "-o", output_file])
    if result != 0:
        raise ValueError("Error converting file to markdown with pandoc")

    # Read the file and delete temporary files
    with open(output_file, "r") as f:
        markdown = f.read()
    os.remove(output_file)
    os.remove(filename)

    return markdown


def convert(input_file, filename) -> str:
    plain_text_filetypes = [
        ".txt",
        ".csv",
        ".tsv",
        ".md",
        ".yaml",
        ".toml",
        ".json",
        ".json5",
        ".jsonc",
    ]
    # Already a plain text file that wouldn't benefit from pandoc so return the content
    if any(filename.endswith(ft) for ft in plain_text_filetypes):
        with open(input_file, "r") as f:
            return f.read()

    if filename.endswith(".pdf"):
        return convert_pdf(input_file)

    return convert_pandoc(input_file, filename)


def chunk_to_length(text, max_length=512):
    chunks = []
    while len(text) > max_length:
        chunks.append(text[:max_length])
        text = text[max_length:]
    chunks.append(text)
    return chunks


def predict(queries, documents, document_filenames, max_characters) -> list[list[str]]:
    queries = queries.split("\n")
    document_filenames = document_filenames.split("\n")

    # Convert the documents to text
    converted_docs = [
        convert(doc, filename) for doc, filename in zip(documents, document_filenames)
    ]

    # Return if the total length is less than the max characters
    total_doc_lengths = sum([len(doc) for doc in converted_docs])
    if total_doc_lengths < max_characters:
        return [[doc] for doc, _ in converted_docs]

    # Embed the documents in 512 character chunks
    chunked_docs = [chunk_to_length(doc, 512) for doc in converted_docs]
    embedded_docs = [embed(queries, chunks) for chunks in chunked_docs]

    # Get a structure like {query: [(doc_idx, chunk_idx, score), (doc_idx, chunk_idx, score), ...]}
    query_embeddings = {}
    for doc_idx, embedded_doc in enumerate(embedded_docs):
        for query, doc_scores in embedded_doc.items():
            doc_scores_with_doc = [
                (doc_idx, chunk_idx, score) for (chunk_idx, score) in doc_scores
            ]
            if query not in query_embeddings:
                query_embeddings[query] = []
            query_embeddings[query] = query_embeddings[query] + doc_scores_with_doc

    # Sort the embeddings by score
    for query, doc_scores in query_embeddings.items():
        query_embeddings[query] = sorted(doc_scores, key=lambda x: x[2], reverse=True)

    # Choose the top embedding from each query until we reach the max characters
    # Getting a structure like [[chunk, ...]]
    document_embeddings = [[] for _ in range(len(documents))]
    total_chars = 0
    while (
        total_chars < max_characters
        and sum([len(x) for x in query_embeddings.values()]) > 0
    ):
        for query, doc_scores in query_embeddings.items():
            if len(doc_scores) == 0:
                continue

            # Grab the top score for the query
            doc_idx, chunk_idx, _ = doc_scores.pop(0)

            # Ensure we have space
            chunk = chunked_docs[doc_idx][chunk_idx]
            if total_chars + len(chunk) > max_characters:
                continue

            # Ensure we haven't already added this chunk from this document
            if chunk_idx in document_embeddings[doc_idx]:
                continue

            # Add the chunk
            document_embeddings[doc_idx].append(chunk_idx)
            total_chars += len(chunk)

    # Get the actual text for the chunks
    document_embeddings = [
        [chunked_docs[doc_idx][chunk_idx] for chunk_idx in chunks]
        for doc_idx, chunks in enumerate(document_embeddings)
    ]

    return document_embeddings