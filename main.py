from fastapi import FastAPI


app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Now work can be done here!!"}

