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
    version="0.6.0-SNAPSHOT",
    description="A python SDK for submarine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apache/submarine",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "six>=1.10.0",
        "numpy==1.18.5",
        "pandas",
        "sqlalchemy",
        "sqlparse",
        "pymysql",
        "requests==2.26.0",
        "urllib3>=1.15.1",
        "certifi>=14.05.14",
        "python-dateutil>=2.5.3",
        "pyarrow==0.17.0",
        "mlflow>=1.15.0",
        "boto3>=1.17.58",
    ],
    extras_require={
        "tf": ["tensorflow>=1.14.0,<2.0.0"],
        "tf-latest": ["tensorflow"],
        "pytorch": ["torch>=1.5.0", "torchvision>=0.6.0"],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    license="Apache License, Version 2.0",
    maintainer="Apache Submarine Community",
    maintainer_email="dev@submarine.apache.org",
)
