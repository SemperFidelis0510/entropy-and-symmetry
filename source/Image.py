import os.path


class Image:
    def __init__(self, rawData, path):
        self.path = path
        self.rawData = rawData
        self.preprocessedData = None
        self.transformedData = {}
        self.entropyResults = []
        self.size = os.path.getsize(path)
        self.pixel_size = rawData.size
    def load(self):
        pass

    def display(self):
        pass