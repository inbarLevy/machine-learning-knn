from tkinter import *
import json

with open('knn_result.json') as json_file:
    knn_result = json.load(json_file)


acc = 0
optimal_k = []
for k in knn_result:
    if knn_result[k] > acc:
        optimal_k = [k]
        acc = knn_result[k]
    elif knn_result[k] == acc:
        optimal_k.append(k)
print('Best K: {}, Best Accuracy: {}'.format(optimal_k, acc))

window = Tk()


def show_acc():
    text['text'] = 'The accuracy of k={} is: {} %'.format(var.get(), round(knn_result[str(var.get())] * 100, 3))


def create_button(key):
    return Radiobutton(window, text='k= ' + str(key), font=['david', 13], bg='white', variable=var, value=key, padx=13,
                       command=show_acc)


window.configure(bg='white')
window.title('KNN Accuracy')
labelName = Label(window, text='Choose k:', bg='white', relief=FLAT, font=['david', 15]).grid(row=0, columnspan=4)

var = IntVar()
text = Label(window, text='', fg='blue', bg='white', relief=FLAT, font=['david', 15])
text.grid(row=6, rowspan=2, columnspan=4)

numOfCols = 5
rowCount = 1
knnList = list(knn_result.keys())

while knnList:
    for i in range(1, numOfCols + 1):
        if knnList:
            nextKey = knnList.pop(0)
            nextButton = create_button(nextKey).grid(row=rowCount, column=i - 1)
    rowCount += 1

window.mainloop()

