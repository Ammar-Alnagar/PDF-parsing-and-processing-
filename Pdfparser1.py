import subprocess
import os
import shutil
import string
import random
from pypdf import PdfReader
import ocrmypdf


def random_word(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def convert_pdf(input_file):
    reader = PdfReader(input_file)
    metadata = extract_metadata_from_pdf(reader)
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

    return text, metadata


def extract_text_from_pdf(reader):
    full_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if len(text) > 0:
            full_text += f"---- Page {idx} ----\n" + page.extract_text() + "\n\n"

    return full_text.strip()


def extract_metadata_from_pdf(reader):
    return {
        "author": reader.metadata.author,
        "creator": reader.metadata.creator,
        "producer": reader.metadata.producer,
        "subject": reader.metadata.subject,
        "title": reader.metadata.title,
    }


def convert_pandoc(input_file, filename):
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


def convert(input_file, filename):
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
            return f.read(), {}

    if filename.endswith(".pdf"):
        return convert_pdf(input_file)

    return convert_pandoc(input_file, filename), {}
