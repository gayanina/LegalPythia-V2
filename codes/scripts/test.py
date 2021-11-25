import pdfplumber 
import docx2txt

def pdf_to_text(file):
    pdf = pdfplumber.open(file)
    page = pdf.pages[0]
    text = page.extract_text()
    pdf.close()
    return text

def doc_to_text(file):
    text = docx2txt.process(file)
    text = text.replace('\n\n',' ')
    text = text.replace('  ',' ')
    return text.encode('utf8')

# if 'pdf' in 'testset_hypothesis_6.pdf': 
#     print('hello')
# else:
#     print()

strn = pdf_to_text('/Users/gayanin/RA-Work/Legal Pythia/testset_premise_6.pdf')
strn1 = doc_to_text('/Users/gayanin/RA-Work/Legal Pythia/testset_hypothesis_6.docx')

print(strn1)