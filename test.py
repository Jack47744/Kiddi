import pickle
filename = 'random_forest.sav'
clf = pickle.load(open(filename, 'rb'))
print(clf)