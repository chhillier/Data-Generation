import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# 1. Define the input EPUB and the output text file
epub_path = 'your_book.epub'
output_txt_path = 'book_content.txt'

def epub_to_text(epub_path):
    """
    Extracts text content from an EPUB file.
    """
    book = epub.read_epub(epub_path)
    full_text = ""
    
    # Iterate through all items in the EPUB (HTML files, images, etc.)
    for item in book.get_items():
        # We only want the HTML documents
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Get the raw HTML content
            html_content = item.get_content()
            
            # Use BeautifulSoup to parse the HTML and get only the text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the body of the HTML and extract its text
            body = soup.find('body')
            if body:
                text = body.get_text(separator='\n', strip=True)
                full_text += text + "\n\n"
                
    return full_text

# 3. Run the conversion and save the output
print(f"Extracting text from '{epub_path}'...")
extracted_text = epub_to_text(epub_path)

with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write(extracted_text)
    
print(f"Text extraction complete. Content saved to '{output_txt_path}'")