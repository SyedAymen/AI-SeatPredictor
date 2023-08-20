import pickle
from fastapi import FastAPI
app = FastAPI()

model = pickle.load(open('aiseatpredictor_decisionTree.pkl', 'rb'))


@app.get("/{rank}/{caste}")
def read_root(rank: int, caste: int):
    global model
    predictions = model.predict([[rank, caste]])
    return {"college": predictions[0][0], "branch": predictions[0][1]}
