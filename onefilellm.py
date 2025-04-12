import requests
from urllib.parse import urlparse
from PyPDF2 import PdfReader
import os
import sys
import tiktoken
import nltk
from nltk.corpus import stopwords
import re
import nbformat
from nbconvert import PythonExporter
import pyperclip
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import xml.etree.ElementTree as ET

EXCLUDED_DIRS = ["dist", "node_modules", ".git", "__pycache__", ".venv"]  # Add any other directories to exclude here

def safe_file_read(filepath, fallback_encoding='latin1'):
    try:
        with open(filepath, "r", encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding=fallback_encoding) as file:
            return file.read()

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def process_ipynb_file(temp_file):
    with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
        notebook_content = f.read()

    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))
    return python_code

def process_local_directory(local_path, output):
    # Chech if local_path is a file
    if os.path.isfile(local_path):
        output.write(f"# {'-' * 3}\n")
        output.write(f"# Filename: {local_path}\n")
        output.write(f"# {'-' * 3}\n\n")

        if local_path.endswith(".ipynb"):
            output.write(process_ipynb_file(local_path))
        else:
            with open(local_path, "r", encoding='utf-8', errors='ignore') as f:
                output.write(f.read())

        output.write("\n\n")
        return
        

    for root, dirs, files in os.walk(local_path):
        # Modify dirs in-place to exclude specified directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            if is_allowed_filetype(file):
                print(f"Processing {os.path.join(root, file)}...")

                output.write(f"# {'-' * 3}\n")
                output.write(f"# Filename: {os.path.join(root, file)}\n")
                output.write(f"# {'-' * 3}\n\n")

                file_path = os.path.join(root, file)

                if file.endswith(".ipynb"):
                    output.write(process_ipynb_file(file_path))
                else:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        output.write(f.read())

                output.write("\n\n")

def process_local_folder(local_path):
    def process_local_directory(local_path):
        content = [f'<source type="local_directory" path="{escape_xml(local_path)}">']

        if os.path.isfile(local_path):
            content.append(f'<file name="{escape_xml(local_path)}">')

            if local_path.endswith(".ipynb"):
                content.append(escape_xml(process_ipynb_file(local_path)))
            else:
                with open(local_path, "r", encoding='utf-8', errors='ignore') as f:
                    content.append(escape_xml(f.read()))

            content.append('</file>')
        else:
            for root, dirs, files in os.walk(local_path):
                # Exclude directories
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

                for file in files:
                    if is_allowed_filetype(file):
                        print(f"Processing {os.path.join(root, file)}...")

                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, local_path)
                        content.append(f'<file name="{escape_xml(relative_path)}">')

                        if file.endswith(".ipynb"):
                            content.append(escape_xml(process_ipynb_file(file_path)))
                        else:
                            with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                                content.append(escape_xml(f.read()))

                        content.append('</file>')
        content.append('</source>')
        return '\n'.join(content)

    formatted_content = process_local_directory(local_path)
    print("All files processed.")
    return formatted_content

def process_arxiv_pdf(arxiv_abs_url):
    pdf_url = arxiv_abs_url.replace("/abs/", "/pdf/") + ".pdf"
    response = requests.get(pdf_url)
    pdf_content = response.content

    with open('temp.pdf', 'wb') as pdf_file:
        pdf_file.write(pdf_content)

    text = []
    with open('temp.pdf', 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            text.append(pdf_reader.pages[page].extract_text())

    formatted_text = f'<source type="arxiv_paper" url="{arxiv_abs_url}">\n'
    formatted_text += '<paper>\n'
    formatted_text += escape_xml(' '.join(text))
    formatted_text += '\n</paper>\n'
    formatted_text += '</source>'

    os.remove('temp.pdf')
    print("ArXiv paper processed successfully.")

    return formatted_text

def preprocess_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as input_file:
        input_text = input_file.read()

    def process_text(text):
        text = re.sub(r"[\n\r]+", "\n", text)
        # Update the following line to include apostrophes and quotation marks
        text = re.sub(r"[^a-zA-Z0-9\s_.,!?:;@#$%^&*()+\-=[\]{}|\\<>`~'\"/]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    try:
        # Try to parse the input as XML
        root = ET.fromstring(input_text)

        # Process text content while preserving XML structure
        for elem in root.iter():
            if elem.text:
                elem.text = process_text(elem.text)
            if elem.tail:
                elem.tail = process_text(elem.tail)

        # Write the processed XML to the output file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print("Text preprocessing completed with XML structure preserved.")
    except ET.ParseError:
        # If XML parsing fails, process the text without preserving XML structure
        processed_text = process_text(input_text)
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(processed_text)
        print("XML parsing failed. Text preprocessing completed without XML structure.")

def get_token_count(text, disallowed_special=[], chunk_size=1000):
    enc = tiktoken.get_encoding("cl100k_base")

    # Remove XML tags
    text_without_tags = re.sub(r'<[^>]+>', '', text)

    # Split the text into smaller chunks
    chunks = [text_without_tags[i:i+chunk_size] for i in range(0, len(text_without_tags), chunk_size)]
    total_tokens = 0

    for chunk in chunks:
        tokens = enc.encode(chunk, disallowed_special=disallowed_special)
        total_tokens += len(tokens)
    
    return total_tokens  
    
def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        # Remove the following lines to stop converting apostrophes and quotes
        # .replace("\"", "&quot;")
        # .replace("'", "&apos;")
    )

def is_excluded_file(filename):
    """
    Check if a file should be excluded based on patterns.

    Args:
        filename (str): The file path to check

    Returns:
        bool: True if the file should be excluded, False otherwise
    """
    excluded_patterns = [
        '.pb.go',  # Proto generated Go files
        '_grpc.pb.go',  # gRPC generated Go files
        'mock_',  # Mock files
        '/generated/',  # Generated files in a generated directory
        '/mocks/',  # Mock files in a mocks directory
        '.gen.',  # Generated files with .gen. in name
        '_generated.',  # Generated files with _generated in name
    ]

    return any(pattern in filename for pattern in excluded_patterns)

def is_allowed_filetype(filename):
    """
    Check if a file should be processed based on its extension and exclusion patterns.

    Args:
        filename (str): The file path to check

    Returns:
        bool: True if the file should be processed, False otherwise
    """
    # First check if file matches any exclusion patterns
    if is_excluded_file(filename):
        return False

    # Then check if it has an allowed extension
    allowed_extensions = [
        '.go',
        '.proto',
        '.py',
        '.txt',
        '.md',
        '.cjs',
        '.html',
        '.json',
        '.ipynb',
        '.h',
        '.localhost',
        '.example'
    ]

    return any(filename.endswith(ext) for ext in allowed_extensions)

def main():
    console = Console()

    intro_text = Text("\nInput Paths or URLs Processed:\n", style="dodger_blue1")
    input_types = [
        ("• Local folder path (flattens all files into text)", "bright_white"),
        ("• ArXiv Paper URL", "bright_white"),
    ]

    for input_type, color in input_types:
        intro_text.append(f"\n{input_type}", style=color)

    intro_panel = Panel(
        intro_text,
        expand=False,
        border_style="bold",
        title="[bright_white]Copy to File and Clipboard[/bright_white]",
        title_align="center",
        padding=(1, 1),
    )
    console.print(intro_panel)

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = Prompt.ask("\n[bold dodger_blue1]Enter the path or URL[/bold dodger_blue1]", console=console)
    
    console.print(f"\n[bold bright_green]You entered:[/bold bright_green] [bold bright_yellow]{input_path}[/bold bright_yellow]\n")

    input_paths = input_path.split(",")

    output_file = "uncompressed_output.txt"
    processed_file = "compressed_output.txt"
    urls_list_file = "processed_urls.txt"

    with Progress(
        TextColumn("[bold bright_blue]{task.description}"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("[bright_blue]Processing...", total=100)

        try:
            outputs = []
            for i, input_path in enumerate(input_paths):
                input_path = input_path.strip()
                outputs.append(process_local_folder(input_path))

            final_output = "\n".join(["<body>"] + outputs + ["</body>"])

            progress.update(task, advance=50)

            # Write the uncompressed output
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(final_output)


            # Process the compressed output
            preprocess_text(output_file, processed_file)

            progress.update(task, advance=50)

            compressed_text = safe_file_read(processed_file)
            compressed_token_count = get_token_count(compressed_text)
            console.print(f"\n[bright_green]Compressed Token Count:[/bright_green] [bold bright_cyan]{compressed_token_count}[/bold bright_cyan]")

            uncompressed_text = safe_file_read(output_file)
            uncompressed_token_count = get_token_count(uncompressed_text)
            console.print(f"[bright_green]Uncompressed Token Count:[/bright_green] [bold bright_cyan]{uncompressed_token_count}[/bold bright_cyan]")

            console.print(f"\n[bold bright_yellow]{processed_file}[/bold bright_yellow] and [bold bright_blue]{output_file}[/bold bright_blue] have been created in the working directory.")

            pyperclip.copy(uncompressed_text)
            console.print(f"\n[bright_white]The contents of [bold bright_blue]{output_file}[/bold bright_blue] have been copied to the clipboard.[/bright_white]")

        except Exception as e:
            console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
            console.print("\nPlease check your input and try again.")
            raise  # Re-raise the exception for debugging purposes
        
if __name__ == "__main__":
    main()
