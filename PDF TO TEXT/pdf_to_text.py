import pdfplumber

# 1. Define the input PDF and the output text file
pdf_path = '/Users/craig/Downloads/Understanding-Digital-Signal-Processing.pdf'
output_txt_path = 'PDF TO TEXT/text_files/Understanding_Digital_Signal_Processing.txt'

# 2. Open the PDF and extract text page by page
full_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        # The .extract_text() method is very good at preserving layout
        text = page.extract_text()
        if text:
            full_text += text + "\n\n" # Add a separator between pages

# 3. Save the extracted text to a file
with open(output_txt_path, 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"Text extraction complete. Content saved to '{output_txt_path}'")