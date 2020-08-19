import csv
import sys
import copy

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    with open(filename) as file:
        reader = csv.reader(file)
        firstRow = True
        labelRow = []
        for row in reader:
            if firstRow:
                firstRow = False
                continue
            element = []
            #0, Administrative, int
            element.append(int(row[0]))
            #1, Administrative_Duration, float
            element.append(float(row[1]))
            #2, Informational, int
            element.append(int(row[2]))
            #3, Informational_Duration, float
            element.append(float(row[3]))
            #4, ProductRelated, int
            element.append(int(row[4]))
            #5, ProductRelated_Duration, float
            element.append(float(row[5]))
            #6, BounceRates, float
            element.append(float(row[6]))
            #7, ExitRates, float
            element.append(float(row[7]))
            #8, PageValues, float
            element.append(float(row[8]))
            #9, SpecialDay, float
            element.append(float(row[9]))
            #10, Month, int
            Month = None
            m = row[10]
            if m == 'Jan':
                Month = 0
            elif m == 'Feb':
                Month = 1
            elif m == 'Mar':
                Month = 2
            elif m == 'Apr':
                Month = 3
            elif m == 'May':
                Month = 4
            elif m == 'June':
                Month = 5
            elif m == 'Jul':
                Month = 6
            elif m == 'Aug':
                Month = 7
            elif m == 'Sep':
                Month = 8
            elif m == 'Oct':
                Month = 9
            elif m == 'Nov':
                Month = 10
            elif m == 'Dec':
                Month = 11
            element.append(int(Month))
            #11, OperatingSystems, int
            element.append(int(row[11]))
            #12, Browser, int
            element.append(int(row[12]))
            #13, Region, int
            element.append(int(row[13]))
            #14, TrafficType, int
            element.append(int(row[14]))
            #15, VisitorType, int
            VisitorType = None
            vt = row[15]
            if vt == 'Returning_Visitor':
                VisitorType = 1
            else:
                VisitorType = 0
            element.append(int(VisitorType))
            #16, Weekend, int
            Weekend = None
            w = row[16]
            if w == 'TRUE':
                Weekend = 1
            elif w == 'FALSE':
                Weekend = 0
            else:
                raise Exception(w)
            element.append(int(Weekend))
            #17, Revenue, int
            Label = None
            l = row[17]
            if l == 'TRUE':
                Label = 1
            elif l == 'FALSE':
                Label = 0
            labels.append(int(Label))
            evidence.append(copy.deepcopy(element))
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, returns a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    returns a tuple (sensitivity, specificty).

    Assumes each label is either a 1 (positive) or 0 (negative).

    `sensitivity` is a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` is a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    posAcurate = 0
    negAcurate = 0
    total = len(labels)
    numTrue = 0
    numFalse = 0
    for i in range(total):
        if labels[i] == 1:
            if labels[i] == predictions[i]:
                posAcurate += 1
            numTrue += 1
        else:
            if labels[i] == predictions[i]:
                negAcurate += 1
            numFalse += 1

    sensitivity = float(posAcurate/numTrue)
    specificity = float(negAcurate/numFalse)
    return (sensitivity,specificity)


if __name__ == "__main__":
    main()
