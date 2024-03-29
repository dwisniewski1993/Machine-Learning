from QLearning.simpleQLearning import SimpleQLearning
from QLearning.deepQLearning import DeepQLearning


def main():
    # Simple Q Learning Model
    sql = SimpleQLearning()
    sql.train()
    sql.play_best_game()
    sql.close_environment()

    # Deep Q Learning Model
    #dql = DeepQLearning()
    #dql.train()
    #dql.play_games(n_games=3)
    #dql.close_environment()
