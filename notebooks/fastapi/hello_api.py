from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return "Hello, world!"


@app.get("/greet/{name}")
def greet(name: str):
    return f"Hello, {name}!"
