/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

module.exports = {
    docs: [
        {
            Introduction: [],
            "Getting Started": [
                "gettingStarted/localDeployment",
                "gettingStarted/notebook",
                "gettingStarted/python-sdk",
            ],
            "User Docs": [
                {
                    "Submarine on Kubernetes": [
                        "userDocs/k8s/run-experiment-template-rest",
                        "userDocs/k8s/run-pytorch-experiment-rest",
                        "userDocs/k8s/run-tensorflow-experiment-rest",
                        "userDocs/k8s/how-to-use-tensorboard",
                    ],
                },
                {
                    "Submarine on Yarn": [
                        "userDocs/yarn/YARNRuntimeGuide",
                        "userDocs/yarn/TestAndTroubleshooting",
                        "userDocs/yarn/Dockerfiles",
                        "userDocs/yarn/WriteDockerfileMX",
                        "userDocs/yarn/WriteDockerfilePT",
                        "userDocs/yarn/WriteDockerfileTF",
                    ],
                },
                {
                    "Submarine SDK": [
                        "userDocs/submarine-sdk/pysubmarine/development",
                        "userDocs/submarine-sdk/pysubmarine/tracking",
                    ],
                },
                {
                    "Submarine Security": [
                        "userDocs/submarine-security/spark-security/README",
                        "userDocs/submarine-security/spark-security/build-submarine-spark-security-plugin",
                    ],
                },
            ],
            "Administrator Docs": [
                {
                    "Submarine on Kubernetes": [
                        "adminDocs/k8s/README",
                        "adminDocs/k8s/kind",
                        "adminDocs/k8s/helm",
                    ],
                },
                {
                    "Submarine on Yarn": ["adminDocs/yarn/README"],
                },
            ],
            "Developer Docs": [
                "devDocs/README",
                "devDocs/BuildFromCode",
                "devDocs/Development",
                "devDocs/IntegrationTest",
            ],
            Community: [
                "community/README",
                "community/HowToCommit",
                "community/contributing",
            ],
            "Design Docs": [
                "designDocs/architecture-and-requirements",
                "designDocs/implementation-notes",
                "designDocs/environments-implementation",
                "designDocs/experiment-implementation",
                "designDocs/notebook-implementation",
                "designDocs/storage-implementation",
                {
                    "Submarine Server": [
                        "designDocs/submarine-server/architecture",
                        "designDocs/submarine-server/experimentSpec",
                    ],
                },
                {
                    "WIP Design Docs": [
                        "designDocs/wip-designs/submarine-launcher",
                        "designDocs/wip-designs/submarine-clusterServer",
                        "designDocs/wip-designs/security-implementation",
                    ],
                },
            ],
            Releases: [
                "releases/submarine-release-0.2.0",
                "releases/submarine-release-0.3.0",
                "releases/submarine-release-0.4.0",
                "releases/submarine-release-0.5.0",
            ],
            RoadMap: [],
        },
    ],
    api: [
        "api/environment",
        "api/experiment",
        "api/experiment-template",
        "api/notebook",
    ],
};
