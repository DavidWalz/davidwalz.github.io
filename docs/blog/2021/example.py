# %% [markdown]

# I'm playing around with the possibilities of mkdocs and [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter)
# This is an example post from a `.py` file, that is, the source is a plain python file where the cells are indicated in [percentage-format](https://jupytext.readthedocs.io/en/latest/formats.html#the-percent-format).
# Hence, code cells start with `# %%` and markdown cells with `# %% [markdown]`.
# Headers don't work. To get the above title, I had to add it to the `mkdocs.yml`

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# Some more text ...

# %%
x = np.linspace(0, 1)
print(x)

# %% [markdown] # Titles are ignored
# So, it sort of works but without headers and hence without navigation.
# Writing plain markdown files and including code sections feels much more natural and does not come with these disadvantages.
# The only advantage for `.py` files is that they can be executed.

