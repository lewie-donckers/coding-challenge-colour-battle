import numpy as np
import math
import random
import pygame

class World:

    MOVES = {
        'n' : np.array([0, 1], dtype=np.int8),
        'e' : np.array([1, 0], dtype=np.int8),
        'w' : np.array([-1, 0], dtype=np.int8),
        's' : np.array([0, -1], dtype=np.int8),
        'h' : np.array([0, 0], dtype=np.int8)
    }

    def __init__(self):
        self.next_bot_id = 1 # Start at 1. 0 will show empty square
        self.bots = []

    def add_bot(self, bot):
        bot.id = self.next_bot_id
        self.next_bot_id += 1
        self.bots += [bot]

    def setup(self):
        cells_per_bot = 8 * 8
        self.grid_length = math.ceil(math.sqrt(cells_per_bot * len(self.bots)))

        # The grid will contain a colour, designated by some bot's id
        self.grid = np.zeros((self.grid_length, self.grid_length), dtype=np.int8)

        # Setup bots
        self.bot_storage = { }
        for bot in self.bots:
            # Reset storage
            self.bot_storage[bot.id] = { }

            # Random start positions. Bots might start on top of another
            bot.position = np.array([
                random.randint(0, self.grid_length-1), 
                random.randint(0, self.grid_length-1)], dtype=np.int8)

    def determine_new_tile_colour(self, floor_colour, bot_colour):
        if floor_colour == 0: return bot_colour
        return [floor_colour, 0, bot_colour][abs(bot_colour - floor_colour) % 3]

    def step(self):

        # Create enemies list
        enemies = [{"id": bot.id, "position": bot.position} for bot in self.bots]

        # Determine next moves
        for bot in self.bots:
            bot.next_move = bot.determine_next_move(
                {'id':bot.id, 'position': bot.position},
                self.grid, 
                enemies, 
                self.bot_storage[bot.id],
                None)

        # Execute moves after all bots determined what they want to do
        for bot in self.bots:
            bot.position = np.add(bot.position, self.MOVES[bot.next_move])
            bot.position[0] = max(min(bot.position[0], self.grid_length - 1), 0)
            bot.position[1] = max(min(bot.position[1], self.grid_length - 1), 0)

        # Refresh new colours
        occupancy = { }
        for bot in self.bots:
            floor_colour = self.grid[bot.position[1]][bot.position[0]]
            new_colour = 0
            if floor_colour is not 0 and not floor_colour in occupancy:
                # This tile haven't been painted this step
                new_colour = self.determine_new_tile_colour(floor_colour, bot.id)
            self.grid[bot.position[1]][bot.position[0]] = new_colour
            occupancy[bot.id] = True

    def get_score(self):
        return {
            bot.id: np.count_nonzero(self.grid == bot.id)
            for bot in self.bots
        }


