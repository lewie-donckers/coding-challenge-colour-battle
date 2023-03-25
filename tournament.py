import argparse
import numpy as np
import matplotlib.pyplot as plt
from robots.bot_list import BotList
from world import World
import os, time, math, sys, atexit
import threading
import concurrent.futures, multiprocessing

parser = argparse.ArgumentParser(description='Nobleo Colour Run')
parser.add_argument('--rounds', type=int, default=2000,
                    help='number of rounds in a single game')
parser.add_argument('--games', type=int, default=200,
                    help='number of games to play')
parser.add_argument('--threads', type=int, default=5,
                    help='number of games to run simultaniously')
parser.add_argument('--graph', action='store_true',
                    help='Shows a histogram of the bots performance')
args = parser.parse_args()

n_bots = len(BotList)
n_rounds = int(args.rounds)
n_games = int(args.games)
total_entries = n_rounds * n_games

# This is all the metrics that we will collect from all the threads
results_lock = threading.Lock()
scores = np.zeros((n_bots, n_games))
time_min = [0.0] * n_bots
time_avg = [0.0] * n_bots
time_max = [0.0] * n_bots

# This is a thread that will print the progress
# of all the games currently in progress in
# other threads. It will automatically terminate
# once the last game is done
progress_lock = threading.Lock()
progress = { }
def progress_printer():
    games_left = {game: True for game in range(n_games)}
    busy = False

    time_started = time.time()
    total_rounds = n_games * n_rounds
    update_period = 1
    average_bucket = 20
    average_rounds_per_update = 0
    last_rounds_left = total_rounds
    while busy or len(games_left) != 0:
        os.system('clear')

        progress_lock.acquire()
        progress_ = progress.copy()
        progress_lock.release()

        if len(progress_) == 0:
            busy = False
            print("Nothing in progress")
        else:
            busy = True
            for game in progress_.keys():
                games_left.pop(game, None)

            # How much rounds are still left
            rounds_left = len(games_left) * n_rounds
            for game, completion in progress_.items():
                rounds_left += math.floor((1-completion) * n_rounds)
            total_completion = (total_rounds-rounds_left)*100/total_rounds

            # Calculate some kind of estimated time left
            rounds_completed = last_rounds_left - rounds_left
            if average_rounds_per_update == 0:
                average_rounds_per_update = rounds_completed
            else:
                average_rounds_per_update -= average_rounds_per_update / average_bucket
                average_rounds_per_update += rounds_completed / average_bucket
            last_rounds_left = rounds_left
            if average_rounds_per_update == 0:
                time_remaining = math.inf 
            else:  
                time_remaining = round(rounds_left / average_rounds_per_update * update_period)

            # Print the information
            print(f"{'Game':>5} | {round(total_completion,1)}% | Left: {len(games_left)} / {n_games} |  Est. Time Left: {time_remaining} seconds")
            for game, completion in progress_.items():
                bar_length = 40
                filledLength = math.ceil(completion * bar_length)
                bar = '*' * filledLength + '-' * (bar_length - filledLength)
                print(f'{game:>5} |{bar}| {round(completion * 100, 1)}%')
        time.sleep(update_period)
    os.system('clear')
    print(f"Tournament finished in {round(time.time() - time_started)} seconds")
progress_printer_thread = threading.Thread(target=progress_printer, daemon=True)
progress_printer_thread.start()

# This is the function that will execute a single game
# and capture some information about it. It will terminate
# when the game is done
def game_runner(game):
    # We run the game in harsh mode, meaning we don't simply ignore
    # it when your bot crashes or returns an invalid move. Your bot
    # will just sit in the same place. Also includes some checks to 
    # ensure the bots aren't cheating.
    _world = World(harsh=True)
    for bot in BotList:
        _world.add_bot(bot)
    _world.setup(n_rounds)
    
    progress_lock.acquire()
    progress[game] = 0
    progress_lock.release()

    _scores = [0] * n_bots
    _time_min = [0.0] * n_bots
    _time_avg = [0.0] * n_bots
    _time_max = [0.0] * n_bots
    
    # Run the game
    while not _world.step(measure_time=True):
        for index, bot in enumerate(_world.bots):
            _time_min[index] = min(bot.measured_time, _time_min[index])
            _time_avg[index] += bot.measured_time / n_rounds
            _time_max[index] = max(bot.measured_time, _time_max[index])
        
        # Update the value used by the progress printer
        progress_lock.acquire()
        progress[game] = _world.current_round / n_rounds
        progress_lock.release()

    # The game is completed. Clean up
    # and capture some information to return
    progress_lock.acquire()
    progress.pop(game, None)
    progress_lock.release()

    max_score = _world.grid.size
    for bot_id, bot_score in _world.get_score().items():
        # Here we calculate the percentage
        _scores[_world.colour_map[bot_id] - 1] = bot_score / max_score * 100

    # Update all the results
    results_lock.acquire()
    for index in range(n_bots):
        scores[index, game] = _scores[index]
        time_min[index] = min(_time_min[index], time_min[index])
        time_avg[index] += _time_avg[index] / n_games
        time_max[index] = max(_time_max[index], time_max[index])
    results_lock.release()

# Here we will spawn the threads that will run all the games concurrently
# It's quite cool that Python has something like this built in
with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
    future_game = {executor.submit(game_runner, game): game for game in range(n_games)}
    for future in concurrent.futures.as_completed(future_game):
        result = future_game[future]
        future.result()

# Wait for progress printer thread to finish
progress_printer_thread.join()

# We are done with the tournament. Now display some info
# Create a dummy world to get bot names and such
world = World()
for bot in BotList:
    world.add_bot(bot)
world.setup(n_rounds)
total_scores = scores.sum(axis=1) / n_games
def fmt(time):
    return f"{round(time * 1e6, 3)}"

# Calculate the final scores and print 
print("============================================================")
print("The Final Scores")
print("============================================================")
sorted_indexes = list(range(n_bots))
sorted_indexes.sort(reverse=True, key=lambda i: total_scores[i])
print(f"{'Rank':<5}{'Name':<30}{'Contributor':<20}{'Avg [us]':<15}{'Score':<7}")
for rank, i in enumerate(sorted_indexes):
    bot = world.bots[i]
    print(f"{rank+1:<5}{bot.get_name():<30}{bot.get_contributor():<20}{fmt(time_avg[i]):<15}{round(total_scores[i], 3):<7} %")

print("\n\n")

print("============================================================")
print("Best Efficiency")
print("============================================================")
sorted_indexes = list(range(n_bots))
sorted_indexes.sort(reverse=True, key=lambda i: total_scores[i]/time_avg[i])
print(f"{'Rank':<5}{'Name':<30}{'Contributor':<20}{'Avg [us]':<15}{'Score [%]':<12}{'Efficiency [%/us]':<12}")
for rank, i in enumerate(sorted_indexes):
    bot = world.bots[i]
    efficiency = total_scores[i] / (time_avg[i] * 1e6)
    print(f"{rank+1:<5}{bot.get_name():<30}{bot.get_contributor():<20}{fmt(time_avg[i]):<15}{round(total_scores[i], 3):<12}{round(efficiency, 3):<12}")

if args.graph:
    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    for i, bot in enumerate(world.bots):
        y, x = np.histogram(scores[i], bins=math.ceil(scores.max()/2))
        linestyle = "." if i > 16 else "--" if i > 8 else "-"
        plt.plot(x[:-1], y, linestyle=linestyle, label=bot.get_name())
    
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.tight_layout()
    plt.subplots_adjust(left=0.045, bottom=0.08, top=0.95)

    plt.title("Histogram showing scores reached over games")
    plt.ylabel("Quantity")
    plt.xlabel("Score [%]")
    plt.show()
