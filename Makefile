### Set make options ###
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

### Development tasks ###
.PHONY: clean
clean:

.PHONY: train
train:
	pipenv run gradient workflows run --id ${GRADIENT_TRAIN_WORKFLOW_ID} --path ./.gradient/workflows/train.yml

.PHONY: test
test: coverage lint regression

.PHONY: lint regression
lint regression:
	act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04 -j $@

.PHONY: lint-format
lint-format:
	pipenv run black --config .github/linters/.python-black .
	pipenv run isort --sp .github/linters/.isort.cfg .

.PHONY: coverage
coverage:
	pipenv run coverage run --source clmbot -m pytest clmbot
	pipenv run coverage html
