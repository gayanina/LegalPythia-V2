from enum import Flag
import pdfplumber 
import docx2txt
import pandas as pd

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

json_text =[[
    "The",
    "ORG",
    "#0cf"
  ],
  [
    "World",
    "ORG",
    "#0cf"
  ],
  [
    "weekly",
    "DATE",
    "#fd1"
  ],
  [
    "Covid",
    "PERSON",
    "#faa"
  ],
]

ls_org = []
ls_per = []
ls_gpe = []
ls_loc = []
ls_date = []
ls_mon = []
rest = []

for ls in json_text:
    if (ls[1] == 'ORG'):
        ls_org.append(ls[0])
    if (ls[1] == 'PERSON'):
        ls_per.append(ls[0])
    if (ls[1] == 'GPE'):
        ls_gpe.append(ls[0])
    if (ls[1] == 'LOC'):
        ls_loc.append(ls[0])
    if (ls[1] == 'DATE'):
        ls_date.append(ls[0])
    if (ls[1] == 'MONEY'):
        ls_mon.append(ls[0])
    else:
        rest.append(ls)

print(ls_org)
print(ls_per)
print(ls_loc)
print(ls_gpe)
print(rest)

print("ORG - Organizations")
print(*ls_org, sep=', ')
print("\n")
print("PER - Persons")
print(*ls_per, sep=', ')
print("\n")
print("GPE - Organizations")
print(*ls_gpe, sep=', ')
print("\n")
print("LOC - Locations")
print(*ls_loc, sep=', ')
print("\n")
print("DATE - Date and Time")
print(*ls_date, sep=', ')
print("\n")
print("MONEY - Monetary Values")
print(*ls_mon, sep=', ')


def in_list(list_of_lists, item, ls_):

    FLAG = 0
    for list_ in list_of_lists:
        if item in list_:
            FLAG =1
    if (FLAG ==1):
      print(item)
      print(*ls_, sep=', ')


in_list(json_text, "ORG", ls_org)
# in_list(json_text, "PERSON", ls_per)