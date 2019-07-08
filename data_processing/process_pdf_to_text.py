# takes the raw testing data from carefuson and creates and saves a term document matrix
from os import listdir
from os.path import isfile, join
from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


class pdf_to_text(object):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        all_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

        for f in all_files:
            self.convert_pdf_to_txt(f)

    def convert_pdf_to_txt(self, f):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = "utf-8"
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = file(self.input_dir + "/" + f, "rb")
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(
            fp,
            pagenos,
            maxpages=maxpages,
            password=password,
            caching=caching,
            check_extractable=True,
        ):
            interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()

        # saves the text
        with open(self.output_dir + "/" + f.replace("pdf", "txt"), "w") as writer:
            writer.write(text)

        print("file create")

