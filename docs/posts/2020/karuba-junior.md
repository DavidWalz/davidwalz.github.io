# Karuba Junior

[Karuba Junior](https://www.boardgamegeek.com/boardgame/234439/karuba-junior) is a cooperative tile-placement adventure game.
Having lost twice in a row I wanted to calculate the odds of winning this game.

![](/images/karuba.jpg#medium)

As expected for a child games the rules are dead simple. 
The players take turns in drawing a card and using it to extend or end one of the 4 starting roads.
Tigers and treasures end a road.
Forks and crossroads add one and two additional roads, respectively, as long as they are placed in such a way that none of the roads are blocked by another card, which in practice is always possible.
The game is won if all 3 treasures are found.
The game is lost if there is no open road left, or if the pirates advanced a total of 9 fields, which happens by drawing the corresponding pirate cards. 

Let's find the odds of winning the game through Monte Carlo. 
We need 3 counters: the number of treasures found, the number of open roads, and the number of pirate moves.
For each card we define how the counters change in form of a 3-component vector. 
Then we can accumulate the changes in the random order they are drawn and determine which win/loss condition occurs first.

There are 28 cards:
* 3 treasures
* 3 tigers
* 11 straight and curved roads
* 3 forks
* 1 crossroads
* 6 pirate cards: 3 cards with one movement point, 2 two's and 1 three


```python
import numpy as np

# card = (#treasure, #roads, #pirates)
cards = np.concatenate([
    np.repeat([[1, -1, 0]], 3, axis=0),  # treasure
    np.repeat([[0, -1, 0]], 3, axis=0),  # tiger
    np.repeat([[0, 0, 0]], 11, axis=0),  # simple road
    np.repeat([[0, 1, 0]], 4, axis=0),  # fork
    np.repeat([[0, 2, 0]], 1, axis=0),  # crossroad
    np.repeat([[0, 0, 1]], 3, axis=0),  # pirate 1
    np.repeat([[0, 0, 2]], 2, axis=0),  # pirate 2
    np.repeat([[0, 0, 3]], 1, axis=0),  # pirate 3
])

def simulate():
    """Simulate a game and determine the win or loss condition"""
    np.random.shuffle(cards)
    
    # all counter start from 0
    (treasures, roads, pirates) = cards.cumsum(axis=0).T
    
    # round when all 3 treasures found
    i_treasure = np.where(treasures == 3)[0][0]
    
    # round when pirates arrive at the beach
    i_pirates = np.where(pirates >= 9)[0][0]
        
    # check if all roads are blocked
    if (roads == -4).any():
        i_roads = np.where(roads <= -4)[0][0]
    else:
        i_roads = np.inf
    
    # note: the case that the third treasure also closes the last road is correctly registered as a win
    return np.argmin([i_treasure, i_roads, i_pirates])
```


```python
n = 100000
res = [simulate() for i in range(n)]
frequency = np.bincount(res) / n
```


```python
print('Probability of outcomes')
print(f'Win:                  p={frequency[0]:.3f}')
print(f'Loss (roads blocked): p={frequency[1]:.3f}')
print(f'Loss (pirates):       p={frequency[2]:.3f}')
```

    Probability of outcomes
    Win:                  p=0.508
    Loss (roads blocked): p=0.052
    Loss (pirates):       p=0.441


So we have a ~50% win probability. Pirates are the most likely reason for losing. Losing due to blocked roads happens rarely if played correctly, but this is a game for 4-8 year olds after all.
