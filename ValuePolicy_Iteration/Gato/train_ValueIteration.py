from ValueIteration import ValueIteration
import json
from tqdm import tqdm


agent = ValueIteration()
agent.value_iteration()

policy = {}
for x in agent.policy:
    policy[str(x)] = agent.policy[x]

V = {}

for x in agent.V:
    V[str(x)] = agent.V[x]

with open("ValuePolicy_Iteration/Values/ValueIteration_VALUE_TTT.json", 'w') as json_file:
    json.dump(V, json_file)

with open("ValuePolicy_Iteration/Policies/ValueIteration_POLICY_TTT.json", 'w') as json_file:
    json.dump(policy, json_file)

