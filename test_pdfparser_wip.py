import os
import sys
import time
import zipfile
import base64
import re
import uuid
import pymupdf
import logging
from pathlib import Path

# Import MinerU library
from magic_pdf.libs.hash_utils import compute_sha256
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.tools.common import do_parse, prepare_env


def read_fn(path):
    """Read file from disk."""
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)


def parse_pdf(doc_path, output_dir, end_page_id, is_ocr=False, layout_mode='layoutlmv3', 
              formula_enable=True, table_enable=False, language=''):
    """Parse PDF and convert to Markdown."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
        pdf_data = read_fn(doc_path)
        parse_method = "ocr" if is_ocr else "auto"
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
            layout_model=layout_mode,
            formula_enable=formula_enable,
            table_enable=table_enable,
            lang=language,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(f"Error parsing PDF: {e}")
        return None, None


def compress_directory_to_zip(directory_path, output_zip_path):
    """Compress a directory to a ZIP file."""
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(f"Error compressing directory: {e}")
        return -1


def image_to_base64(image_path):
    """Convert image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    """Replace image links with base64 encoded images."""
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"

    return re.sub(pattern, replace, markdown_text)


def convert_pdf_to_markdown(file_path, output_dir='./output', max_pages=5, is_ocr=False, 
                             layout_mode='layoutlmv3', formula_enable=True, 
                             table_enable=False, language=''):
    """Main conversion function."""
    # Convert to PDF if not already a PDF
    file_path = to_pdf(file_path)

    # Parse PDF
    local_md_dir, file_name = parse_pdf(
        file_path, 
        output_dir, 
        max_pages - 1, 
        is_ocr, 
        layout_mode, 
        formula_enable, 
        table_enable, 
        language
    )

    if not local_md_dir or not file_name:
        logger.error("Failed to parse PDF")
        return None

    # Compress directory
    archive_zip_path = os.path.join(output_dir, compute_sha256(local_md_dir) + ".zip")
    zip_result = compress_directory_to_zip(local_md_dir, archive_zip_path)
    
    if zip_result != 0:
        logger.error("Failed to create ZIP archive")

    # Read markdown file
    md_path = os.path.join(local_md_dir, file_name + ".md")
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    
    # Replace image links with base64
    md_content = replace_image_with_base64(txt_content, local_md_dir)

    # Get layout PDF path
    new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")

    return {
        'markdown_content': md_content,
        'original_text': txt_content,
        'zip_archive_path': archive_zip_path,
        'layout_pdf_path': new_pdf_path
    }


def to_pdf(file_path):
    """Convert file to PDF if not already a PDF."""
    with pymupdf.open(file_path) as f:
        if f.is_pdf:
            return file_path
        else:
            pdf_bytes = f.convert_to_pdf()
            unique_filename = f"{uuid.uuid4()}.pdf"
            tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

            with open(tmp_file_path, 'wb') as tmp_pdf_file:
                tmp_pdf_file.write(pdf_bytes)

            return tmp_file_path


def main():
    """Command-line interface for PDF to Markdown conversion."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert PDF to Markdown')
    parser.add_argument('input_file', help='Path to input PDF file')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--max-pages', type=int, default=5, help='Maximum pages to convert')
    parser.add_argument('--ocr', action='store_true', help='Force OCR')
    parser.add_argument('--layout-mode', default='layoutlmv3', choices=['layoutlmv3', 'doclayout_yolo'], help='Layout model')
    parser.add_argument('--no-formula', dest='formula_enable', action='store_false', help='Disable formula recognition')
    parser.add_argument('--table', action='store_true', help='Enable table recognition')
    parser.add_argument('--language', default='', help='Language for parsing')

    args = parser.parse_args()

    # Perform conversion
    result = convert_pdf_to_markdown(
        file_path=args.input_file,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        is_ocr=args.ocr,
        layout_mode=args.layout_mode,
        formula_enable=args.formula_enable,
        table_enable=args.table,
        language=args.language
    )

    if result:
        logger.info(f"Markdown saved to: {os.path.join(args.output_dir, 'result.md')}")
        logger.info(f"Zip archive saved to: {result['zip_archive_path']}")
        logger.info(f"Layout PDF saved to: {result['layout_pdf_path']}")

        # Optional: Save markdown to file
        with open(os.path.join(args.output_dir, 'result.md'), 'w', encoding='utf-8') as f:
            f.write(result['markdown_content'])


if __name__ == "__main__":
    main()