# dynaplex_toyproject

A multi-node supply chain simulation that trains a PPO reinforcement learning agent (for now) to minimise inventory costs (holding, backlog, and ordering).

## How to Run

**1. Install dependencies**
```bash
./dynaplex/bin/pip install -r requirements.txt
```

**2. Train and simulate**
```bash
python main.py
```

This runs a baseline simulation first, then trains a PPO agent and runs a second simulation with the learned policy.

**Disclaimer**

CODE IN PLOTS.PY AND RECORD.PY ARE NOT MY OWN. THEY HAVE BEEN GENERATED / TAKEN FROM A VARIETY OF ONLINE SOURCES. THE REMAINDER OF THE PROJECT IS OF MY OWN 
MAKING. 
```
