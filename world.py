import numpy as np
import math
import random
import time
from robots.bot_control import Move

class World:

    class GameInfo:
        def __init__(self, length, size):
            self.number_of_rounds = length
            self.grid_size = size
            self.current_round = 0

    MOVE_TO_VECTOR = {
        Move.UP: np.array([0, 1],  dtype=np.int16),
        Move.RIGHT: np.array([1, 0],  dtype=np.int16),
        Move.LEFT: np.array([-1, 0], dtype=np.int16),
        Move.DOWN: np.array([0, -1], dtype=np.int16),
        Move.STAY: np.array([0, 0],  dtype=np.int16)
    }

    def __init__(self, harsh=False):
        self.bot_types = []
        self.bots = []
        self.colour_map = {0: 0} # 0 is always zero (white)
        self.harsh = harsh

    def add_bot(self, bot):
        # Remember the type, so we can recreate it every round
        # So that player can store values in the object itself
        self.bot_types += [bot]

    def setup(self, number_of_rounds):
        self.current_round = 0

        # Create the bots
        self.bots = [bot_type() for bot_type in self.bot_types]

        # Create the grid
        cells_per_bot = 8 * 8
        self.grid_length = math.ceil(math.sqrt(cells_per_bot * len(self.bots)))
        self.game_info = self.GameInfo(number_of_rounds, self.grid_length)

        # The grid will contain a colour, designated by some bot's id
        self.grid = np.zeros((self.grid_length, self.grid_length), dtype=np.int16)

        # Randomize a list of IDs that we will assign to bots
        randomized_ids = list(range(1, len(self.bots) + 1))
        random.shuffle(randomized_ids)

        # Give all bots their starting postions and assign IDs   
        for i, bot in enumerate(self.bots):
            # Assign random id (starting at 1)
            bot.id = randomized_ids[i]

            # Assign bot unique number dependent on order added, and
            # not the randomized ID so that colours can remain consistent
            self.colour_map[bot.id] = i + 1

            # Random start positions. Bots might start on top of another
            bot.position = np.array([
                random.randint(0, self.grid_length-1), 
                random.randint(0, self.grid_length-1)], dtype=np.int16)

    def determine_new_tile_colour(self, floor_colour, bot_colour) -> int:
        if floor_colour == 0: return bot_colour
        return [floor_colour, 0, bot_colour][(bot_colour - floor_colour) % 3]

    def step(self, measure_time=False) -> bool:
        # Update game info
        self.current_round += 1
        self.game_info.current_round = self.current_round

        # Create enemies list
        enemies = [{"id": bot.id, "position": bot.position} for bot in self.bots]

        # Determine next moves
        # Do this in a random sequence every time. This will give better
        # time measurements, as found by Jorik.
        for bot in random.sample(self.bots, len(self.bots)):
            start_time = time.time() if measure_time else 0
            try:
                bot.next_move = bot.determine_next_move(self.grid, enemies, self.game_info)
            except Exception as e:
                if self.harsh:
                    # Then we don't care if your bot crashes. We will ignore it.
                    bot.next_move = Move.STAY
                else:
                    raise e
            if measure_time: bot.measured_time = time.time() - start_time
            if not bot.next_move in self.MOVE_TO_VECTOR:
                if self.harsh:
                    # We don't care if you returned the wrong thing.
                    bot.next_move = Move.STAY
                else:
                    raise Exception(f"Bot \"{bot.get_name()}\" attempted an invalid move \"{bot.next_move}\"")

        # Execute moves after all bots determined what they want to do
        for bot in self.bots:
            bot.position = np.add(bot.position, self.MOVE_TO_VECTOR[bot.next_move])
            bot.position[0] = max(min(bot.position[0], self.grid_length - 1), 0)
            bot.position[1] = max(min(bot.position[1], self.grid_length - 1), 0)

        # Paint new colours
        occupancy = np.zeros_like(self.grid)
        for bot in self.bots:
            floor_colour = self.grid[bot.position[1], bot.position[0]]
            new_colour = 0
            if occupancy[bot.position[1], bot.position[0]] == 0:
                # This tile haven't been painted this step
                new_colour = self.determine_new_tile_colour(floor_colour, bot.id)
            self.grid[bot.position[1]][bot.position[0]] = new_colour
            occupancy[bot.position[1], bot.position[0]] += 1

        # Return true if the game is done
        return self.game_info.current_round == self.game_info.number_of_rounds

    def get_score(self):
        return {
            bot.id: np.count_nonzero(self.grid == bot.id)
            for bot in self.bots
        }


