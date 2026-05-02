import base64
import fitz
from markdown_pdf import MarkdownPdf, Section

def get_base64_uri(file_path):
    with open(file_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def build_pdf(md_path, pdf_path):
    # Create the markdown PDF object
    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = "Industry Project Report"
    pdf.meta["author"] = "Raja S"
    
    # 1. Section: Title Page (No TOC)
    title_html = """
    <div style="text-align: left; padding: 40px; font-family: Helvetica, sans-serif;">
        <h1 style="font-size: 32px; margin-bottom: 5px;">Coriolis Technologies</h1>
        <h2 style="font-size: 24px; color: #333; margin-top: 0;">Industry Project Report</h2>
        <br><br><br><br>
        <p style="font-size: 18px; margin-bottom: 0;"><strong>Author:</strong></p>
        <p style="font-size: 18px; margin-top: 0;">Raja S</p>
        <br><br>
        <p style="font-size: 18px; margin-top: 0;">30/04/2026</p>
    </div>
    """
    pdf.add_section(Section(title_html, toc=False))
    
    # 2. Section: Main Content (With TOC)
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Clean up any residual HTML title blocks in the markdown file itself
    if md_content.strip().startswith("<div"):
        # Split on the page break div and take the second part
        parts = md_content.split('page-break-after: always;"></div>')
        if len(parts) > 1:
            md_content = parts[1].strip()

    # Inline the images
    loss_uri = get_base64_uri('loss_plot.png')
    md_content = md_content.replace('![Training Loss for Epoch 1](loss_plot.png)',
                                    f'<img src="{loss_uri}" alt="Training Loss for Epoch 1" width="600"/>')

    arch_uri = get_base64_uri('architecture_diagram.png')
    md_content = md_content.replace('![Architecture Diagram](architecture_diagram.png)',
                                    f'<img src="{arch_uri}" alt="Architecture Diagram" width="600"/>')

    # Add content section and generate preliminary PDF
    pdf.add_section(Section(md_content, toc=True))
    pdf.save(pdf_path)
    
    # 3. Post-process the PDF with PyMuPDF to add headers (like the template)
    doc = fitz.open(pdf_path)
    for i in range(1, len(doc)):
        page = doc[i]
        # Coordinates for header
        rect = fitz.Rect(50, 30, page.rect.width - 50, 50)
        
        # Add the header text
        header_text = "Industry Project Report                                                                       Raja S"
        page.insert_textbox(rect, header_text, fontsize=10, fontname="helv", color=(0.4, 0.4, 0.4))
        
        # Add a subtle line under the header
        page.draw_line(fitz.Point(50, 45), fitz.Point(page.rect.width - 50, 45), color=(0.7, 0.7, 0.7), width=0.5)

    doc.saveIncr()
    doc.close()
    print(f"Saved formatted template PDF to {pdf_path}")

build_pdf('LLM_Distillation_Project_Report.md', 'LLM_Distillation_Project_Report_v7.pdf')
