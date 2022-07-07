import json


class ConfigMan:
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f:
            self.config = json.load(f)

    def get(self, p1, p2, p3=None):
        try:
            if p3 is None:
                return self.config[p1][p2]
            else:
                return self.config[p1][p2][p3]
        except IOError as e:
            print("Cannot find the property")
            print(e)
            quit(-1)
