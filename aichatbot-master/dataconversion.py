import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import textwrap
import re

def clean_text(text):
    # Replace newlines, tabs, and remove strange characters
    text = str(text).replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII chars
    return text.strip()

# Load CSV
csv_path = "D:\\project\\aichatbot\\Train_data.csv"
df = pd.read_csv(csv_path)

# PDF Setup
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

# Title
pdf.set_font("Helvetica", style="B", size=14)
pdf.cell(200, 10, text="Disease Dataset", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
pdf.ln(5)
pdf.set_font("Helvetica", size=11)

# Write data
for idx, row in df.iterrows():
    for col in df.columns:
        content = clean_text(f"{col}: {row[col]}")
        # Wrap very long lines manually (80 chars per line)
        wrapped_lines = textwrap.wrap(content, width=80)
        for line in wrapped_lines:
            pdf.multi_cell(0, 8, text=line)
    pdf.ln(4)

# Save
pdf_path = "D:\\project\\aichatbot\\Train_data.pdf"
pdf.output(pdf_path)
print(f"✅ PDF generated successfully at {pdf_path}")
