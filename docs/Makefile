SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Insight
SOURCEDIR     = meta
BUILDDIR      = build
GITHUBPAGESBRANCH = gh-pages


.PHONY: help clean html github_deploy

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html           to make standalone HTML files"
	@echo "  clean          to clean up build files"
	@echo "  github_deploy  to build and push HTML docs to GitHub Pages"

clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILDDIR)/*

html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

github_deploy: html
	@echo "Deploying to GitHub Pages..."
	# Navigate to the build HTML directory
	cd $(BUILDDIR)/html && \
	# Initialize git (if needed), add, commit, and push changes to gh-pages branch
	git add . && \
	git commit -m "Update documentation" && \
	git push -f origin $(GITHUBPAGESBRANCH)
	@echo "Documentation deployed to GitHub Pages."
