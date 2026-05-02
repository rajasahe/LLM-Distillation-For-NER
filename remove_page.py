from PyPDF2 import PdfReader, PdfWriter

reader = PdfReader("LLM_Distillation_Project_Report.pdf")
writer = PdfWriter()

# Skip the first page (index 0), add the rest
for i in range(1, len(reader.pages)):
    writer.add_page(reader.pages[i])

with open("LLM_Distillation_Project_Report.pdf", "wb") as f:
    writer.write(f)

print("First page removed successfully!")
