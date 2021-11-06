# Moving to MkDocs

In the past I was using [fastpages](https://github.com/fastai/fastpages) for some blogging while on partnal leave with my second daughter. 
While the looks are certainly amazing, I was unconfortable with the amount of magic going on and the lack of control. 
That's why I have decided to switch over to mkdocs / mkdocs-material which I'm already using extensively at work.

Through [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) I'll be able to keep the existing posts in `.ipynb` format. 

### Todos
- [x] Move all .ipynb files and clean of fastpages-specific magic
- [ ] Get the bibliography working, see https://github.com/shyamd/mkdocs-bibtex/
- [x] Make a nice intro page, see e.g. https://ddrscott.github.io/blog/2018/move-to-mkdocs/
- [x] Hide the `In`/`Out` fields inside the notebooks, see https://github.com/danielfrg/mkdocs-jupyter/issues/30
- [ ] Collapse jupyter cells, see https://jupytext.readthedocs.io/en/latest/formats.html#metadata-filtering
- [ ] Adding tags, see https://squidfunk.github.io/mkdocs-material/setup/setting-up-tags/#adding-a-tags-index
- [ ] Redo the logo