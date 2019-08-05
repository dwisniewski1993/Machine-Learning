from QLearning.simpleQLearning import SimpleQLearning


def main():
    sql = SimpleQLearning()
    #sql.train()
    sql.play_best_game()
    sql.close_enviroment()
