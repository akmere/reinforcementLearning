from fastapi import FastAPI
from pydantic import BaseModel, Field, HttpUrl
from liba import *
import json
from numpyencoder import NumpyEncoder

class State(BaseModel):
    x: int
    y: int

class ValueIterationItem(BaseModel):
    states: list[State] = []
    terminal_states: list[State] = []
    rewards: list[float] = []
    actions: list[str] = []
    probs: list[list[float]] = []
    transitional_probs: list[list[list[float]]] = []
    v: list[float] = []
    theta: float | None = 0.01
    gamma: float | None = 0.9

    class Config:
        arbitrary_types_allowed = True


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/items/")
async def create_item(item: ValueIterationItem):
    states = np.array(item.states)
    terminal_states = np.array(item.terminal_states)
    rewards = np.array(item.rewards)
    actions = np.array(item.actions)
    probs = np.array(item.probs)
    transitional_probs = np.array(item.transitional_probs)
    v = np.array(item.v)
    theta = item.theta
    gamma = item.gamma
    iterate_policy(states, terminal_states, rewards, actions, probs, transitional_probs, v, theta, gamma)
    return {"v": json.dumps(v, cls=NumpyEncoder),"probs": json.dumps(probs, cls=NumpyEncoder)}
