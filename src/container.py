class container(object):
    def __init__(self):
        self.container = []
        return

    def getLength(self):
        return self.container.__len__()

    def addBlock(self, newData):
        self.container.append(newData)
        return

    def sortFeatures(self):
        self.container = sorted(self.container, key=lambda x:(x[1], x[2]))
        return

    def printAllContainer(self):
        for index in range(0, self.container.__len__()):
            print('\t',self.container[index])
        return

    def printContainer(self, count):
        print("\tElement's index:", self.container.__len__())
        if count > self.container.__len__():
            self.printAllContainer()
        else:
            for index in range(0, count):
                print('\t', self.container[index])
        return
