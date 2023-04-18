from fastapi import FastAPI
import pickle
import sklearn

app = FastAPI()

def predict_result():
    pickled_model = pickle.load(open('svm_model.pkl', 'rb'))

    # Update si1 based on your dataset features
    si1 = [[63,1,3,145,233,1,0,150,0,1]]

    result = pickled_model.predict(si1)

    s = "Still trying to predict"

    if result[0] == 0.0:
        s = "Congrats! Less Chances of Heart Attack."
    else:
        s = "Sorry! More Chances of Heart Attack."

    return s

@app.get("/")
async def root():
    s = predict_result()
    return s