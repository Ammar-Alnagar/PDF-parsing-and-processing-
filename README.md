# Document Parser

This repository provides a Python-based tool for parsing various types of documents. It extracts text and metadata from PDF files and other plain-text file formats. It can also convert documents into Markdown format using Pandoc.

---

## Features

- Extracts text and metadata from PDF files, with optional OCR for image-based PDFs.
- Supports various plain-text file types, including `.txt`, `.csv`, `.json`, and more.
- Converts non-PDF files into Markdown using Pandoc.
- Handles metadata extraction for PDF files (e.g., author, title, subject).

---



## Prerequisites



Before using the tool, ensure the following dependencies are installed:

- **Python Libraries**
  - `pypdf`
  - `ocrmypdf`

Install these using pip:

```bash
pip install pypdf ocrmypdf

Pandoc

Install Pandoc from official website or via your system's package manager:

sudo apt install pandoc  # On Debian/Ubuntu
brew install pandoc      # On macOS







---





Usage

Function Overview

1. convert_pdf(input_file)

Extracts text and metadata from a PDF file.

Performs OCR for image-based PDFs if necessary.

Returns extracted text and metadata.



2. convert_pandoc(input_file, filename)

Converts non-PDF files to Markdown using Pandoc.

Returns converted Markdown text.



3. convert(input_file, filename)

Main function to handle different file types.

Returns extracted or converted content and metadata.




Example Usage

from document_parser import convert

# File to process
input_file = "example.pdf"
filename = "example.pdf"

# Parse the document
content, metadata = convert(input_file, filename)

print("Content:", content)
print("Metadata:", metadata)


---

Supported File Types

Plain Text Files:

.txt, .csv, .tsv, .md, .json, .yaml, .toml, .json5, .jsonc


PDF Files

Extracts text and metadata, with OCR support for image-based PDFs.


Other Formats

Converts to Markdown using Pandoc.






