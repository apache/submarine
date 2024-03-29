<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Submarine Website

This website is built using [Docusaurus 2](https://v2.docusaurus.io/), a modern static website generator.

## Installation

```shell
# We recommend using node 18(LTS). 
yarn install
```

## Local Development

```shell
yarn start
```

This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

If you want to switch to zh-cn, you can use the following command to start.

```shell
yarn start -- --locale zh-cn
```

## Build

```shell
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

```shell
GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
