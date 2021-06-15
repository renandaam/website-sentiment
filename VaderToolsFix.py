from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import time
import csv

analyzer = SentimentIntensityAnalyzer()

pos_count = 0
pos_correct = 0
count = 0

neg_count = 0
neg_correct = 0

csvFile = open("TranslateDataVaksin2021x.csv", 'w', encoding='utf-8')
csvWriter = csv.writer(csvFile)

with open("TranslateDataVaksinCovidX.csv","r") as f:
    for line in f.read().split('\n'):
        vs = analyzer.polarity_scores(line)['compound']
        # print(line)
        # print(analyzer.polarity_scores(line))
        count += 1


        if vs >= 0 : 
            # print("Positive")
            pos_correct += 1
            pos_count +=1
            csvWriter.writerow([line, "Positive"])
        else : 
            # print("Negative")
            neg_correct += 1
            neg_count +=1
            csvWriter.writerow([line, "Negative"])
    csvWriter = csv.writer(csvFile)
csvFile.close()
        # if not vs['neg'] > 0:
        #     if vs['pos']-vs['neg'] > 0:
        #         print("Positive")
        #         pos_correct += 1
        #     pos_count +=1

        # if not vs['pos'] > 0.1:
        #     if vs['pos']-vs['neg'] <= 0:
        #         print("Negative")
        #         neg_correct += 1
        #     neg_count +=1




# with open("TranslasiDataVaksin.csv","r") as f:
#     for line in f.read().split('\n'):
#         vs = analyzer.polarity_scores(line)
#         if not vs['pos'] > 0:
#             if vs['pos']-vs['neg'] < 0:
#                 neg_correct += 1
#             neg_count +=1


print("Positive accuracy = {}% via {} samples".format(pos_correct/count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/count*100.0, neg_count))