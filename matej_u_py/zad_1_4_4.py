#Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ song.txt.
#Potrebno je napraviti rjecnik koji kao klju ˇ ceve koristi sve razli ˇ cite rije ˇ ci koje se pojavljuju u ˇ
#datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (klju ˇ c) pojavljuje u datoteci. ˇ
#Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.


lyrics = open("song (2).txt")

word_appereance = {}


for line in lyrics:
    line = line.lower()
    words = line.split()
    for word in  words:
        if word in word_appereance:
            number = word_appereance[word]
            number = number + 1
            word_appereance[word] = number
        else:
            word_appereance[word] = 1

print(word_appereance)    