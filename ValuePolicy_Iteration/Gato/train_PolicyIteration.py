from PolicyIteration import PolicyIteration
import json
from tqdm import tqdm

agent = PolicyIteration()
agent.policy_iteration()

policy = {}
for x in agent.policy:
    policy[str(x)] = agent.policy[x]

V = {}

for x in agent.V:
    V[str(x)] = agent.V[x]

with open("ValuePolicy_Iteration/Values/PolicyIteration_VALUE_TTT.json", 'w') as json_file:
    json.dump(V, json_file)

with open("ValuePolicy_Iteration/Policies/PolicyIteration_POLICY_TTT.json", 'w') as json_file:
    policy_json = {str(state): int(agent.policy[state]) if agent.policy[state] != None else None for state in agent.policy}
    json.dump(policy_json, json_file)