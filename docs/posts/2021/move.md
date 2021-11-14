# Moving to MkDocs

In the past I was using [fastpages](https://github.com/fastai/fastpages) for some blogging while on partnal leave with my second daughter. 
While the looks are certainly amazing, I was unconfortable with the amount of magic going on and the lack of control. 
That's why I have decided to switch over to mkdocs / mkdocs-material which I'm already using extensively at work.

Through [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) I'll be able to keep the existing posts in `.ipynb` format. 

With [mkdocs-bibtex](https://github.com/shyamd/mkdocs-bibtex/) I will be able to keep the current bibliograpy, but will have to modify the citation style from `{% cite XYZ %}` to `[@XYZ]`.
See [@Schulz2018] for an example.
This works in markdown files, but not in jupyter notebooks.#
Let's see what can be done here short of switching from mkdocs to jupyter-book.

I'm really missing the graphical overview of the landing page in fastpages which shows the latest posts in order together with the headline and optional a graphic. See [here]((https://fastpages.fast.ai/)) for an example.
Placing and styling images is surprisingly difficult in markdown.
I spent quite some time searching for ways to implement this without customizing the mkdocs theme, without success so far.

### Todos
- [x] Move all .ipynb files and clean of fastpages-specific magic
- [ ] Get the bibliography working, see https://github.com/shyamd/mkdocs-bibtex/
- [x] Make a (nice) intro page, see e.g. https://ddrscott.github.io/blog/2018/move-to-mkdocs/
- [x] Hide the `In`/`Out` fields inside the notebooks, see https://github.com/danielfrg/mkdocs-jupyter/issues/30
- [ ] Collapse jupyter cells, see https://jupytext.readthedocs.io/en/latest/formats.html#metadata-filtering
- [ ] Adding tags, see https://squidfunk.github.io/mkdocs-material/setup/setting-up-tags/#adding-a-tags-index

### References
\bibliography