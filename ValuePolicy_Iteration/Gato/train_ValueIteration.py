from ValueIteration import ValueIteration
import json
from tqdm import tqdm


agent = ValueIteration()
agent.value_iteration()

policy = {}
for x in agent.policy:
    policy[str(x)] = agent.policy[x]

with open("ValuePolicy_Iteration/Policies/valueIterationGato.json", 'w') as json_file:
    json.dump(policy, json_file)

