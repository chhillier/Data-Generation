
# Nougat OCR: PDF Scraping Guide

This guide provides the terminal commands to install and use the Nougat OCR tool to convert complex PDFs (with math and tables) into plain text files.

-----

## 1\. Installation

1.  Open your Terminal.

2.  Activate your Python virtual environment:

    ```bash
    source .venv/bin/activate
    ```

3.  Install the official `nougat-ocr` package using pip:

    ```bash
    pip install nougat-ocr
    ```

4.  (Optional) It's always good practice to keep your pip installer up to date:

    ```bash
    pip install --upgrade pip
    ```

-----

## 2\. Running Nougat

1.  In your terminal, navigate to the directory that contains the PDF you want to process.

    ```bash
    cd /path/to/your/pdfs
    ```

2.  Run the `nougat` command on your PDF file:

    ```bash
    nougat "your_book_name.pdf" -o "output_folder_name"
    ```

      * Replace `"your_book_name.pdf"` with the name of your file.
      * The `-o` flag tells Nougat to create a new folder (e.g., `"output_folder_name"`) to store the results.

-----

## 3\. Using the Output

After the command finishes, you will have a new folder containing one or more `.mmd` (Markdown) files. These are the plain text versions of your PDF pages, with all the math preserved.

You can then move these `.mmd` files into your agent's `knowledge_base` directory to be indexed into its memory.