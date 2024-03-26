# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="apache-submarine",
    version="0.8.0.dev",
    description="A python SDK for submarine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apache/submarine",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"submarine.cli.config": ["cli_config.yaml"]},
    install_requires=[
        "numpy",
        "pandas",
        "sqlalchemy>=1.4.0, <2.0.0",
        "sqlparse",
        "pymysql",
        "requests>=2.26.0",  # SUBMARINE-922. avoid GPL dependency.
        "urllib3>=1.15.1",
        "certifi>=14.05.14",
        "python-dateutil>=2.5.3",
        "pyarrow>=6.0.1",
        "boto3>=1.17.58",
        "click>=8.1.0",
        "rich",
        "dacite",
        "pyaml",
    ],
    extras_require={
        "tf2": [
            "tensorflow>=2.12.0,<2.15.0",
            "numpy>=1.14.5",
            "keras>=2.6.0",
            "tensorflow-addons>=0.17.0",
            "tensorflow-estimator>=2.12.0,<2.15.0",
            "tf_slim==1.1.0",
            # todo(cdmikechen): Based on SUBMARINE-1372, typeguard has recently been upgraded to version 3.0,
            #                   which will restrict some python syntax and types more tightly.
            #                   We are not upgrading this in submarine 0.8.0 for now,
            #                   and will fix version compatibility issues in 0.8.1 or 0.9.0.
            "typeguard<3.0.0",
        ],
        "pytorch": ["torch>=1.5.0", "torchvision>=0.6.0"],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "submarine = submarine.cli.main:entry_point",
        ],
    },
    license="Apache License, Version 2.0",
    maintainer="Apache Submarine Community",
    maintainer_email="dev@submarine.apache.org",
)
