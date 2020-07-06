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

## Python Development
This page provides general Python development guidelines and source build instructions
### Prerequisites
This is required for developing & testing changes, we recommend installing pysubmarine
in its own conda environment by running the following
```bash
conda create --name submarine-dev python=3.6
conda activate submarine-dev

# lint-requirements.txt and test-requirements.txt are in ./submarine-sdk/pysubmarine/github-actions
pip install -r lint-requirements.txt
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
- Use [yapf](https://github.com/google/yapf) to format Python code
- yapf style is configured in `.style.yapf` file
- To autoformat code
```bash
./submarine-sdk/pysubmarine/github-actions/auto-format.sh
```
- Verify linter pass before submitting a pull request by running:
```bash
./submarine-sdk/pysubmarine/github-actions/lint.sh
```
### Unit Testing
We are using [pytest](https://docs.pytest.org/en/latest/) to develop our unit test suite.
After building the project (see below) you can run its unit tests like so:
```bash
cd submarine-sdk/pysubmarine
pytest --cov=submarine -vs
```
### Generate python SDK from swagger
We use [swagger-codegen](https://swagger.io/docs/open-source-tools/swagger-codegen/)
to generate pysubmarine client API that used to communicate with submarine server.

If change below files, please run `./dev-support/pysubmarine/gen-sdk.sh`
to generate latest version of SDK.
- [Bootstrap.java](https://github.com/apache/submarine/blob/master/submarine-server/server-core/src/main/java/org/apache/submarine/server/Bootstrap.java)
- [ExperimentRestApi.java](https://github.com/apache/submarine/blob/master/submarine-server/server-core/src/main/java/org/apache/submarine/server/rest/ExperimentRestApi.java)

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
