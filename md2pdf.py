from markdown_pdf import MarkdownPdf, Section
import sys

def convert_md_to_pdf(md_path, pdf_path):
    pdf = MarkdownPdf(toc_level=2)
    # Removed title section so there is no extra first page
    
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    pdf.add_section(Section(md_content, toc=False))
    pdf.save(pdf_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input.md> <output.pdf>")
        sys.exit(1)
    
    convert_md_to_pdf(sys.argv[1], sys.argv[2])
