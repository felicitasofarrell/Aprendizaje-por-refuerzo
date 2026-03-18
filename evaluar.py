from connect4 import Connect4
from agentes import RandomAgent, DefenderAgent, Greedy2Agent, HeuristicAgent
from principal import TrainedAgent

def jugar(agent1, agent2, n=300, render=False):
    res = {0:0, 1:0, 2:0}
    for _ in range(n):
        game = Connect4(agent1=agent1, agent2=agent2)
        w = game.play(render=render)
        res[w] += 1
    return res

if __name__ == "__main__":
    model = "trained_agent_model.pth"
    trained = TrainedAgent(model_path=model, state_shape=(6,7), n_actions=7, device="cpu")

    print("Vs Random (trained juega primero):", jugar(trained, RandomAgent("Rand"), n=400))
    print("Vs Random (trained juega segundo):", jugar(RandomAgent("Rand"), trained, n=400))

    print("Vs Defender (trained juega primero):", jugar(trained, DefenderAgent("Def"), n=400))
    print("Vs Defender (trained juega segundo):", jugar(DefenderAgent("Def"), trained, n=400))

    print("Vs Greedy (trained juega primero):", jugar(trained, Greedy2Agent("Greedy"), n=400))
    print("Vs Greedy (trained juega segundo):", jugar(Greedy2Agent("Greedy"), trained, n=400))

    print("Vs Heuristic (trained juega primero):", jugar(trained, HeuristicAgent("Heuristic"), n=400))
    print("Vs Heuristic (trained juega segundo):", jugar(HeuristicAgent("Heuristic"), trained, n=400))
