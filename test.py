from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def check(true, pred):
    print (f'F1 Score: {f1_score(true, pred, average="weighted")}')
    print (f'Accuracy: {accuracy_score(true, pred)}')

    return accuracy_score(true, pred)

def test_model(model, X_test, y_test):
    true, pred = [], []
    for z in range (len(y_test)):
        prediction = model.predict(np.expand_dims(X_test[z], 0))
        prediction = int(np.argmax(prediction, axis=1)[0])+1
        true.append(y_test[z])
        pred.append(str(prediction))
    
    return check(true, pred)
