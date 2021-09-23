### Set make options ###
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

### Development tasks ###
.PHONY: build
build:
	docker build -t clmbot .

.PHONY: clean
clean:
	docker rmi clmbot

.PHONY: train
train:
	pipenv run gradient workflows run \
		--id ${GRADIENT_TRAIN_WORKFLOW_ID} \
		--path ./.gradient/workflows/train.yml

.PHONY: fetch
fetch: path = ${HOME}/.clmbot/model/
fetch:
	@mkdir -p "$(path)"
	@declare -a files=( config.json merges.txt pytorch_model.bin special_tokens_map.json tokenizer_config.json tokenizer.json vocab.json ); \
    for file in "$${files[@]}" ; do \
		pipenv run gradient datasets files get \
			--id "${GRADIENT_MODEL_DATASET_ID}:latest" \
			--source-path "$${file}" \
			--target-path "$(path)$${file}"; \
    done

.PHONY: deploy cli
deploy cli:
	python -m clmbot $@

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
