"""Stage 3 (cont.) — open the DOCX in Word, populate the TOC and page-number fields,
save, and export the PDF.
"""
import win32com.client as win32

BASE = r"C:\Users\krisg\Desktop\Drone to 3d mesh\output\386b1914-edb6-4fc6-a685-e6ad7c337225"
DOCX = BASE + r"\stage3\Drone-to-3D-Walkable-World-Project-Report.docx"
PDF = BASE + r"\stage3\Drone-to-3D-Walkable-World-Project-Report.pdf"

wdExportFormatPDF = 17

word = win32.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(DOCX, ReadOnly=False, AddToRecentFiles=False)
    try:
        doc.Repaginate()
        for i in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents.Item(i).Update()
        doc.Fields.Update()
        # second pass: the populated TOC changes pagination, so page numbers must settle
        doc.Repaginate()
        for i in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents.Item(i).Update()
        doc.SaveAs2(DOCX)
        doc.ExportAsFixedFormat(OutputFileName=PDF, ExportFormat=wdExportFormatPDF,
                                OpenAfterExport=False)
        print("pages =", doc.ComputeStatistics(2))  # wdStatisticPages
    finally:
        doc.Close(False)
finally:
    word.Quit()
print("PDF written:", PDF)
