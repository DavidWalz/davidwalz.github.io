# First Orchard

First Orchard is simplified version of the children's board game "Orchard" and is meant as a first game for ages 2+.
The goal of this cooperative game is to collect all 4 fruits from each of 4 fruit trees (red apples, green apples, yellow pears and blue plums) before a crow arrives at the orchard to steal the fruits.
Players take turns to roll a 6-sided dice with the faces showing the 4 different fruit colors, the crow and a fruit basket.
When rolling a color, a fruit of the corresponding type can be collected if there is still one remaining.
When rolling the basket, the player can choose to pick any of the remaining fruits.
When rolling the crow, the crow advances one step. 
The game is lost if the crow advanced a total of 6 steps before all fruits are collected.

![](/images/first-orchard.jpg#medium)

There are children's games that proceed without any decisions by the players.
First Orchard does have one strategic element, which is the choice of the fruit when rolling the basket. 
For optimal play one should always take the fruit type, of which the most fruits remain, so that the chance of rolling a completely harvested fruit is minimized. 
For the lowest win probability, one can do the opposite by trying to completely strip of one fruit tree before moving to the next. 
From experience it's rather futile to explain to a 2-year-old why the latter is not a good strategy, so I wanted to at least know how big the impact is on the win probability.

As this game produces a variable length random sequence, there is no elegant way of vectorizing the calculation. Instead, we simply write out the logic and jit-compile with numba.


```python
from numba import jit
import numpy as np


@jit(nopython=True)
def choose_best_strategy(counter):
    """Pick the type of fruits of which we have the least."""
    return np.argmin(counter[1:]) + 1


@jit(nopython=True)
def choose_worst_strategy(counter):
    """Pick the type of fruit of which we already have the most (but less than 4)."""
    valid_choices = np.arange(1, 5)[counter[1:] < 4]
    return valid_choices[np.argmax(counter[valid_choices])]


@jit(nopython=True)
def simulate(choose=choose_best_strategy):
    """Simulate a single game and return whether we won or not."""
    counter = np.zeros(5)  # [crow, fruit1, fruit2, fruit3, fruit4]
    while (counter[1:] < 4).any():
        i = np.random.randint(0, 6)  # 0: crow, 1 - 4: fruit1-4, 5: choose a fruit
        if i == 5:
            i = choose(counter)  # choose 
        counter[i] += 1
        if counter[0] == 6:  # loss
            return False
    return True  # win
```

With optimal play we have a ~77% win probability. 


```python
p_win = np.mean([simulate(choose_best_strategy) for i in range(100000)])
print(f"{p_win * 100:.1f} %")
```

    76.9 %


The worst play leads to 7% lower win probility. Any possible strategy will land us in between. So now we know.


```python
p_win = np.mean([simulate(choose_worst_strategy) for i in range(100000)])
print(f"{p_win * 100:.1f} %")
```

    70.1 %


Finally, let's look at how much speed we gained from the jit-compilation.


```python
%timeit simulate(choose_worst_strategy)
```

    31 µs ± 1.1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


and without jit


```python
def simulate_nojit(choose=choose_best_strategy):

    def choose_worst_strategy(counter):
        valid_choices = np.arange(1, 5)[counter[1:] < 4]
        return valid_choices[np.argmax(counter[valid_choices])]

    counter = np.zeros(5)  # [crow, fruit1, fruit2, fruit3, fruit4]
    while (counter[1:] < 4).any():
        i = np.random.randint(0, 6)  # 0: crow, 1 - 4: fruit1-4, 5: choose a fruit
        if i == 5:
            i = choose_worst_strategy(counter)
        counter[i] += 1
        if counter[0] == 6:  # loss
            return False
    return True  # win

%timeit simulate_nojit()
```

    385 µs ± 23.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


We gained a factor of >10 which, when running 100000 simulations as done above, makes all the difference between what feels like an instant response and half an eternity :).
This is really nice because the only thing we needed to do was putting a `@jit` decorator in front of the function.


