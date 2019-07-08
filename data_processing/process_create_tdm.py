# takes the raw testing data from carefuson and creates and saves a term document matrix
from os import listdir
import pandas as pd
import textmining


class create_tdm(object):
    def __init__(self, path, subset=None):
        self.path = path

        # determins if it was passed a file or directory
        if path[len(path) - 3 :] == "csv":
            res = self.single_file()
        else:
            res = self.multiple_files()

        tdm = res[0]

        # converst the tdm to a list
        data_tdm = []
        for r in tdm.rows(cutoff=1):
            data_tdm.append(r)

        # converts the list to a data frame and does a little cleanup
        data_tdm = pd.DataFrame(data_tdm)
        data_tdm.columns = data_tdm.ix[0, :]
        data_tdm = data_tdm.drop(0)

        # adds in the class and x_raw
        data_tdm = data_tdm.reset_index(drop=True)
        data_tdm["x_raw"] = res[1]
        data_tdm["_y_"] = res[2]

        self.data_tdm = data_tdm

        # subsets the data set if need be
        if subset is not None:
            print("Subsetting")
            self.data_tdm = self.data_tdm.sample(subset)

        print("TDM created")

    # function for reading documents from directories
    def multiple_files(self):

        # creates the tdm
        data_tdm = []
        tdm = textmining.TermDocumentMatrix()
        y = []
        x_raw = []

        # loops through the folders
        folders = [f for f in listdir(self.path)]
        for folder in folders:
            print(folder)
            # loops through the files
            files = [f for f in listdir(self.path + "/" + folder)]
            for _file in files:
                with open(
                    self.path + "/" + folder + "/" + _file, "r", encoding="ISO-8859-1"
                ) as f:
                    text = f.readlines()

                # if it comes as a list this fixes that
                if type(text) == list:
                    text = " ".join(text)

                # removes any whitespace
                text = " ".join(text.split())

                # converts it all to ascii
                text = text.encode("ascii", "ignore")
                text = text.decode("ascii", "ignore")
                text = text.replace(",", "")

                # saves the actuall text and classification
                x_raw.append(text)
                tdm.add_doc(text)
                y.append(folder)

        # returns the resutls
        return (tdm, x_raw, y)

    # function for creating a tdm from a single file
    def single_file(self):

        # reads in the data
        data = pd.read_csv(self.path, encoding="ISO-8859-1")
        # cleans up some text encoding
        for i in range(data.shape[0]):
            if type(data.x[i]) != "unicode":
                data.x[i] = unicode(data.x[i])
            data.x[i] = data.x[i].encode("ascii", errors="ignore")
        self.data = data

        # removes the file name and keeps just the directory
        file_path = self.path.split("/")
        file_path = "/".join(file_path[: len(file_path) - 1])

        # creates the tdm
        data_tdm = []
        tdm = textmining.TermDocumentMatrix()
        x_raw = []
        y = []

        # adds each row to the tdm
        for i in data.index:
            try:
                tdm.add_doc(data.x[i])
                x_raw.append(data.x[i])
                y.append(data.y[i])
            except:
                print("failed import: " + str(data.x[i]))

        # converst the tdm to a list
        for r in tdm.rows(cutoff=1):
            data_tdm.append(r)

        # returns the resutls
        return (tdm, x_raw, y)

    # saves the results
    def to_csv(self, path):
        self.data_tdm.to_csv(path, index=False)
        print("TDM saved")
