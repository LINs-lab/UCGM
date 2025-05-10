from methods.transports.edm import EDM
from methods.transports.linear import Linear
from methods.transports.random import Random
from methods.transports.relinear import ReLinear
from methods.transports.trigflow import TrigFlow
from methods.transports.triglinear import TrigLinear


TRANSPORTS = {
    "EDM": EDM,
    "Linear": Linear,
    "Random": Random,
    "ReLinear": ReLinear,
    "TrigFlow": TrigFlow,
    "TrigLinear": TrigLinear,
}
