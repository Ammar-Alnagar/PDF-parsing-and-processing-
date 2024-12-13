import PyPDF2
from openpyxl import load_workbook
from pptx import Presentation
import io
import re
import zipfile
import xml.etree.ElementTree as ET
import filetype

# Constants
CHUNK_SIZE = 32000

# --- Utility Functions ---

def xml2text(xml):
    """Extracts text from XML data."""
    text = u''
    root = ET.fromstring(xml)
    for child in root.iter():
        text += child.text + " " if child.text is not None else ''
    return text

def clean_text(content):
    """Cleans text content based on the 'clean' parameter."""
    content = content.replace('\n', ' ')
    content = content.replace('\r', ' ')
    content = content.replace('\t', ' ')
    content = re.sub(r'\s+', ' ', content)
    return content


def split_content(content, chunk_size=CHUNK_SIZE):
    """Splits content into chunks of a specified size."""
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunks.append(content[i:i + chunk_size])
    return chunks

# --- Document Reading Functions ---

def extract_text_from_docx(docx_data, clean=True):
    """Extracts text from DOCX files."""
    text = u''
    zipf = zipfile.ZipFile(io.BytesIO(docx_data))

    filelist = zipf.namelist()

    header_xmls = 'word/header[0-9]*.xml'
    for fname in filelist:
        if re.match(header_xmls, fname):
            text += xml2text(zipf.read(fname))

    doc_xml = 'word/document.xml'
    text += xml2text(zipf.read(doc_xml))

    footer_xmls = 'word/footer[0-9]*.xml'
    for fname in filelist:
        if re.match(footer_xmls, fname):
            text += xml2text(zipf.read(fname))

    zipf.close()
    if clean:
        text = clean_text(text)
    return text, len(text)

def extract_text_from_pptx(pptx_data, clean=True):
    """Extracts text from PPT files."""
    text = u''
    zipf = zipfile.ZipFile(io.BytesIO(pptx_data))

    filelist = zipf.namelist()

    # Extract text from slide notes
    notes_xmls = 'ppt/notesSlides/notesSlide[0-9]*.xml'
    for fname in filelist:
        if re.match(notes_xmls, fname):
            text += xml2text(zipf.read(fname))

    # Extract text from slide content (shapes and text boxes)
    slide_xmls = 'ppt/slides/slide[0-9]*.xml'
    for fname in filelist:
        if re.match(slide_xmls, fname):
            text += xml2text(zipf.read(fname))

    zipf.close()
    if clean:
        text = clean_text(text)
    return text, len(text)

def read_document(file_path, clean=True):
    with open(file_path, "rb") as f:
        file_content = f.read()

    kind = filetype.guess(file_content)

    if kind is None:
        mime = "text"
    else:
        mime = kind.mime

    if mime == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            content = ''
            for page in range(len(pdf_reader.pages)):
                content += pdf_reader.pages[page].extract_text()
            if clean:
                content = clean_text(content)
            return content, len(repr(content))
        except Exception as e:
            return f"Error reading PDF: {e}", 0
    elif mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        try:
            wb = load_workbook(io.BytesIO(file_content))
            content = ''
            for sheet in wb.worksheets:
                for row in sheet.rows:
                    for cell in row:
                        if cell.value is not None:
                            content += str(cell.value) + ' '
            if clean:
                content = clean_text(content)
            return content, len(repr(content))
        except Exception as e:
            return f"Error reading XLSX: {e}", 0
    elif mime == "text/plain":
        try:
            content = file_content.decode('utf-8')
            if clean:
                content = clean_text(content)
            return content, len(repr(content))
        except Exception as e:
            return f"Error reading TXT file: {e}", 0
    elif mime == "text/csv":
        try:
            content = file_content.decode('utf-8')
            if clean:
                content = clean_text(content)
            return content, len(repr(content))
        except Exception as e:
            return f"Error reading CSV file: {e}", 0
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            return extract_text_from_docx(file_content, clean)
        except Exception as e:
            return f"Error reading DOCX: {e}", 0
    elif mime == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        try:
            return extract_text_from_pptx(file_content, clean)
        except Exception as e:
            return f"Error reading PPTX: {e}", 0

    else:
        try:
            content = file_content.decode('utf-8')
            if clean:
                content = clean_text(content)
            return content, len(repr(content))
        except Exception as e:
            return f"Error reading file: {e}", 0