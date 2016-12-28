import math

class Movie():
    def __init__(self):
        self.__name = ""
        self.__kissCount = 0
        self.__kickCount = 0
        self.__genre = ""
    def __str__(self):
        return "name:" + self.__name + " kissCount:" + str(self.__kissCount) + " kickCount" + str(self.__kickCount) + " __genre:" + str(self.__genre)
    @property
    def genre(self):
        return self.__genre
    @genre.setter
    def genre(self, genre):
        self.__genre = genre
    @property
    def kissCount(self):
        return self.__kissCount
    @kissCount.setter
    def kissCount(self, kissCount):
        self.__kissCount = kissCount
    @property
    def kickCount(self):
        return self.__kickCount
    @kickCount.setter
    def kickCount(self, kickCount):
        self.__kickCount = kickCount
    @property
    def name(self):
        return self.__name
    @name.setter
    def name(self, name):
        self.__name = name


def getMovieFromLine(line):
    properties = line.split()
    movie = Movie()
    for property in properties :
        key = property.split(":")[0]
        value = property.split(":")[1]
        setattr(movie, key, value)
    return movie


def getDistance(movie, otherMovie):
    kissDistance = math.pow(float(movie.kissCount) - float(otherMovie.kissCount), 2)
    kickDistance = math.pow(float(movie.kickCount) - float(otherMovie.kickCount), 2)
    return math.sqrt(kissDistance + kickDistance)

def getMovieGenreBykNN(movie, movieList):
    genreList = list()
    distanceList = list()
    for otherMovie in movieList:
        distance = getDistance(movie, otherMovie)
        if len(genreList) < 3:
            genreList.append(otherMovie.genre)
            distanceList.append(distance)
            continue
        maxDistance = max(distanceList)
        if maxDistance > distance:
            maxIndex = distanceList.index(maxDistance)
            genreList.pop(maxIndex)
            distanceList.pop(maxIndex)
            genreList.append(otherMovie.genre)
            distanceList.append(distance)
    genreSet = set(genreList)
    resultGenre = ""
    maxCount = 0
    for genre in genreSet:
        count = genreList.count(genre)
        if maxCount < count:
            resultGenre = genre
            maxCount = count
    return resultGenre

k = 3
movieFile = open("movie.txt", "r")
line = movieFile.readline()

movieList = list()

while line :
    movie = getMovieFromLine(line)
    if movie.genre == "" :
        movie.genre = getMovieGenreBykNN(movie, movieList)
    movieList.append(movie)
    line = movieFile.readline()
print("finish")
for movie in movieList:
    print(movie)
