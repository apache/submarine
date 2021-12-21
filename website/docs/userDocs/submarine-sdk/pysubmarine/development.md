---
title: Python SDK Development
---

<!---
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

This page provides general Python development guidelines and source build instructions

### Prerequisites

This is required for developing & testing changes, we recommend installing pysubmarine
in its own conda environment by running the following

```bash
conda create --name submarine-dev python=3.6
conda activate submarine-dev

# Install auto-format and lints (lint-requirements.txt is in ./dev-support/style-check/python)
pip install -r lint-requirements.txt

# Install mypy (mypy-requirements.txt is in ./dev-support/style-check/python)
pip install -r mypy-requirements.txt

# test-requirements.txt is in ./submarine-sdk/pysubmarine/github-actions
pip install -r test-requirements.txt

# Installs pysubmarine from current checkout
pip install ./submarine-sdk/pysubmarine
```

### PySubmarine Docker

We also use docker to provide build environments for CI, development,
generate python sdk from swagger.

```bash
./run-pysubmarine-ci.sh
```

The script does the following things:

- Start an interactive bash session
- Mount submarine directory to /workspace and set it as home
- Switch user to be the same user that calls the `run-pysubmarine-ci.sh`

### Coding Style

- Use [isort](https://github.com/PyCQA/isort) to sort the Python imports and [black](https://github.com/psf/black) to format Python code
- Both style is configured in `pyproject.toml`
- To autoformat code

```bash
./dev-support/style-check/python/auto-format.sh
```

- Use [flake8](https://github.com/PyCQA/flake8) to verify the linter, its' configure is in `.flake8`.
- Also, we are using [mypy](https://github.com/python/mypy) to check the static type in `submarine-sdk/pysubmarine/submarine`.
- Verify linter pass before submitting a pull request by running:

```bash
./dev-support/style-check/python/lint.sh
```

- If you encouter a unexpected format, use the following method
```python
# fmt: off
  "Unexpected format, formated by yourself"
# fmt: on
```

### Unit Testing

We are using [pytest](https://docs.pytest.org/en/latest/) to develop our unit test suite.
After building the project (see below) you can run its unit tests like so:

```bash
cd submarine-sdk/pysubmarine
```

- Run unit test

```shell script
pytest --cov=submarine -vs -m "not e2e"
```

- Run integration test

```shell script
pytest --cov=submarine -vs -m "e2e"
```

> Before run this command in local, you should make sure the submarine server is running.

### Generate python SDK from swagger

We use [open-api generator](https://openapi-generator.tech/docs/installation/#jar)
to generate pysubmarine client API that used to communicate with submarine server.

1. To generate different API Component, please change the code in [Bootstrap.java](https://github.com/apache/submarine/blob/master/submarine-server/server-core/src/main/java/org/apache/submarine/server/Bootstrap.java). If just updating java code for `NotebookRestApi` , `ExperimentRestApi` or `EnvironmentRestApi`, please skip step 1.

    ```java
    SwaggerConfiguration oasConfig = new SwaggerConfiguration()
                .openAPI(oas)
                .resourcePackages(Stream.of("org.apache.submarine.server.rest")
                        .collect(Collectors.toSet()))
                .resourceClasses(Stream.of("org.apache.submarine.server.rest.NotebookRestApi",
                        "org.apache.submarine.server.rest.ExperimentRestApi",
                        "org.apache.submarine.server.rest.EnvironmentRestApi")
                        .collect(Collectors.toSet()));
    ```
    > After starting the server, `http://localhost:8080/v1/openapi.json` will includes API specs for `NotebookRestApi`, `ExperimentRestApi` and `EnvironmentRestApi`


2. [swagger_config.json](https://github.com/apache/submarine/blob/master/dev-support/pysubmarine/swagger_config.json) defines the import path for python SDK

    Ex: 

    For `submarine.client`
    ```json
    {
      "packageName" : "submarine.client",
      "projectName" : "submarine.client",
      "packageVersion": "0.7.0-SNAPSHOT"
    }
    ```

    > Usage: `import submarine.client...`

2. Execute `./dev-support/pysubmarine/gen-sdk.sh` to generate latest version of SDK.

    > Notice: Please install required package before running the script: [lint-requirements.txt](https://github.com/apache/submarine/blob/master/dev-support/style-check/python/lint-requirements.txt)
3. In `submarine/submarine-sdk/pysubmarine/client/api_client.py` line 74

    Please change
    ```python
    "long": int if six.PY3 else long,  # noqa: F821
    ```
    to 
    ```python
    "long": int,
    ```

### Model Management Model Development

For local development, we can access cluster's service easily thanks to [telepresence](https://www.telepresence.io/).
To elaborate, we can develop the sdk in local but can reach out to mlflow server by proxy.

1. Install telepresence follow [the instruction](https://www.telepresence.io/reference/install).
2. Start proxy pod

```
telepresence --new-deployment submarine-dev
```

3. You can develop as if in the cluster.

### Upload package to PyPi

For Apache Submarine committer and PMCs to do a new release.

1. Change the version from 0.x.x-SNAPSHOT to 0.x.x
   in [setup.py](https://github.com/apache/submarine/blob/master/submarine-sdk/pysubmarine/setup.py)
2. Install Python packages

```bash
cd submarine-sdk/pysubmarine
pip install -r github-actions/pypi-requirements.txt
```

3. Compiling Your Package

It will create `build`, `dist`, and `project.egg.info`
in your local directory

```bash
python setup.py bdist_wheel
```

4. Upload python package to TestPyPI for testing

```bash
python -m twine upload --repository testpypi dist/*
```

5. Upload python package to PyPi

```bash
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```
