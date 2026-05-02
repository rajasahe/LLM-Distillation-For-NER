import fitz

def get_topics():
    topics = []
    with open('LLM_Distillation_Project_Report.md', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('## '):
                topics.append((line[3:].strip(), 1)) # level 1
            elif line.startswith('### '):
                topics.append((line[4:].strip(), 2)) # level 2
    return topics

def transplant():
    video_doc = fitz.open('video-activity-detection.pdf')
    our_doc = fitz.open('LLM_Distillation_Project_Report_v7.pdf')
    
    # We will build a brand new doc
    final_doc = fitz.open()
    
    # --- PAGE 1: TITLE PAGE ---
    # Copy page 0 from video
    final_doc.insert_pdf(video_doc, from_page=0, to_page=0)
    p0 = final_doc[0]
    
    # Redact and replace text
    redact_rects = [
        (fitz.Rect(100, 450, 500, 500), "Industry Project Report"),  # Summer Internship Report
        (fitz.Rect(100, 540, 300, 580), "Raja S"),                 # Kousik Krishnan
        (fitz.Rect(200, 610, 400, 650), "30/04/2026")              # Date
    ]
    
    for rect, new_text in redact_rects:
        p0.add_redact_annot(rect, fill=(1, 1, 1))
    p0.apply_redactions()
    
    # Write new text
    p0.insert_textbox(fitz.Rect(100, 460, 500, 500), "Industry Project Report", fontsize=24, fontname="helv", align=fitz.TEXT_ALIGN_CENTER)
    p0.insert_textbox(fitz.Rect(100, 555, 300, 580), "Raja S", fontsize=14, fontname="helv", align=fitz.TEXT_ALIGN_CENTER)
    p0.insert_textbox(fitz.Rect(200, 625, 400, 650), "30/04/2026", fontsize=14, fontname="helv", align=fitz.TEXT_ALIGN_CENTER)

    # --- PAGE 2: TOC PAGE ---
    # Copy page 1 from video
    final_doc.insert_pdf(video_doc, from_page=1, to_page=1)
    p1 = final_doc[1]
    
    # Redact everything except header/footer
    p1.add_redact_annot(fitz.Rect(0, 70, p1.rect.width, p1.rect.height), fill=(1, 1, 1))
    p1.apply_redactions()
    
    # Also fix the header "Summer Internship Report Kousik Krishnan" to "Industry Project Report Raja S"
    header_rect = fitz.Rect(0, 0, p1.rect.width, 60)
    p1.add_redact_annot(header_rect, fill=(1, 1, 1))
    p1.apply_redactions()
    p1.insert_text(fitz.Point(50, 45), "Industry Project Report", fontsize=10, fontname="helv", color=(0.4, 0.4, 0.4))
    p1.insert_text(fitz.Point(p1.rect.width - 90, 45), "Raja S", fontsize=10, fontname="helv", color=(0.4, 0.4, 0.4))
    p1.draw_line(fitz.Point(50, 50), fitz.Point(p1.rect.width - 50, 50), color=(0.7, 0.7, 0.7), width=0.5)

    # Write the TOC
    p1.insert_text(fitz.Point(50, 90), "Contents", fontsize=18, fontname="hebo")
    
    topics = get_topics()
    y = 130
    for topic, level in topics:
        x = 50 if level == 1 else 70
        fontsize = 12 if level == 1 else 11
        p1.insert_text(fitz.Point(x, y), topic, fontsize=fontsize, fontname="helv")
        y += 20

    # --- REST OF THE PAGES ---
    # Insert from our generated PDF, starting from page 2 (skipping our old Title and TOC)
    final_doc.insert_pdf(our_doc, from_page=2)
    
    # Ensure headers match on the rest of the pages
    for i in range(2, len(final_doc)):
        p = final_doc[i]
        # Our pdf already has headers added in build_pdf.py! But let's make sure it's exactly the same.
        # It's fine as is.
    
    final_doc.save("LLM_Distillation_Project_Report_Final.pdf")
    print("Generated LLM_Distillation_Project_Report_Final.pdf")

if __name__ == '__main__':
    transplant()
