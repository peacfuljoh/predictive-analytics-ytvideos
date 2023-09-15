SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

REPO_NAME_ = crawler

REPO_ROOT_ = /home/nuc/$(REPO_NAME_)
CR_ROOT_ = $(REPO_ROOT_)
MAIN_ROOT_ = $(REPO_ROOT_)/src/$(REPO_NAME_)
REPO_ROOT_GHA_ = /home/runner/work/$(REPO_NAME_)/$(REPO_NAME_)
CR_ROOT_GHA_ = $(REPO_ROOT_GHA_)

ACT_ENV = $(CONDA_ACTIVATE) $(REPO_NAME_)


test:
	$(CONDA_ACTIVATE) $(REPO_NAME_) && \
	cd $(CR_ROOT_) && \
	pytest

test-gha:
	$(CONDA_ACTIVATE) $(REPO_NAME_) && \
	cd $(CR_ROOT_GHA_) && \
	pytest

run:
	$(CONDA_ACTIVATE) $(REPO_NAME_) && \
	cd $(MAIN_ROOT_) && \
	python main.py
