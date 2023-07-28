import pickle

class RawDataHandler:
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
