---
title: 下载 Apache Submarine
---

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

Apache Submarine 的最新版本是 `0.8.0`.

- Apache Submarine `0.78.0` 于2023年9月23日发布 ([发布公告](/zh-cn/releases/submarine-release-0.8.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.8.0))
  - 二进制部署包:
    [submarine-dist-0.8.0.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/0.8.0/submarine-dist-0.8.0.tar.gz) (126 MB, [checksum](https://www.apache.org/dist/submarine/0.8.0/submarine-dist-0.8.0.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/0.8.0/submarine-dist-0.8.0.tar.gz.asc))
  - 源代码:
    [apache-submarine-0.8.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/0.8.0/apache-submarine-0.8.0-src.tar.gz) (9.7 MB, [checksum](https://www.apache.org/dist/submarine/0.8.0/apache-submarine-0.8.0-src.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/0.8.0/apache-submarine-0.8.0-src.tar.gz.asc))
  - Docker 镜像:
    - [submarine server](https://hub.docker.com/layers/apache/submarine/server-0.8.0/images/sha256-3885aadbf8e7806c3e9f08f025856d851036e881d617d2b08eaedbe0c92c2fcf) `docker pull apache/submarine:server-0.8.0`
    - [submarine database](https://hub.docker.com/layers/apache/submarine/database-0.8.0/images/sha256-c13466bd95c92e9abdb1d507ce6a57f26ec00e6e75a474d65ead1e3f7e8b34aa) `docker pull apache/submarine:database-0.8.0`
    - [submarine jupyter-notebook](https://hub.docker.com/layers/apache/submarine/jupyter-notebook-0.8.0/images/sha256-44d1e768fa85180ede8b05517f5a9749ce68905147100763a44cca54fa5f25ea) `docker pull apache/submarine:jupyter-notebook-0.8.0`
    - [submarine jupyter-notebook-gpu](https://hub.docker.com/layers/apache/submarine/jupyter-notebook-gpu-0.8.0/images/sha256-8154a754b1741e4667e74bf6532904c5b3e2687e543b8847f549e1392a4ad196) `docker pull apache/submarine:jupyter-notebook-gpu-0.8.0`
    - [submarine quickstart](https://hub.docker.com/layers/apache/submarine/quickstart-0.8.0/images/sha256-45d04388ec03f5111af112eb4cb55faa99e2db0e530fe37785f96ae0195dc9de) `docker pull apache/submarine:quickstart-0.8.0`
    - [submarine mlflow](https://hub.docker.com/layers/apache/submarine/mlflow-0.8.0/images/sha256-a80817973c16a830a5c1ff3258dab7f852820e246b8444d9cdbb800009097b6e) `docker pull apache/submarine:mlflow-0.8.0`
    - [submarine operator](https://hub.docker.com/layers/apache/submarine/operator-0.8.0/images/sha256-e4e055ea2a209a261e72b72dc09f65a932654f9037bac1ebeb521f7d9d84dbf6) `docker pull apache/submarine:operator-0.8.0`
    - [submarine agent](https://hub.docker.com/layers/apache/submarine/agent-0.8.0/images/sha256-68869cd841e0edaf2493bf1886e4d51cb35cad595eb474c4196a5e959997b0a8) `docker pull apache/submarine:agent-0.8.0`

  - SDK:
    - [PySubmarine](https://pypi.org/project/apache-submarine/0.8.0/) `pip install apache-submarine==0.8.0`

## 验证文件完整性

您必须使用 PGP 或 MD5 签名来 [验证](https://www.apache.org/info/verification.html) 下载文件的完整性。 
此签名应与 [KEYS](https://www.apache.org/dist/submarine/KEYS) 文件匹配。

```
gpg --import KEYS
gpg --verify submarine-dist-X.Y.Z-src.tar.gz.asc
```

## 旧版本

- Apache Submarine `0.7.0` 于2022年4月25日发布 ([发布公告](/zh-cn/releases/submarine-release-0.7.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.7.0))
  - 二进制部署包:
    [submarine-dist-0.7.0.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.7.0/submarine-dist-0.7.0.tar.gz) (148 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.7.0/submarine-dist-0.7.0-src.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.7.0/submarine-dist-0.7.0.tar.gz.asc))
  - 源代码:
    [submarine-dist-0.7.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.7.0/submarine-dist-0.7.0-src.tar.gz) (8.3 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.7.0/submarine-dist-0.7.0-src.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.7.0/submarine-dist-0.7.0-src.tar.gz.asc)))
  - Docker 镜像:
    - [submarine server](https://hub.docker.com/layers/submarine/apache/submarine/server-0.7.0/images/sha256-4f9c8e41d9242f2d92f9a9def0b9e43efdd6a9b42e23ae3a1fa25afee48d0370?context=explore) `docker pull apache/submarine:server-0.7.0`
    - [submarine database](https://hub.docker.com/layers/submarine/apache/submarine/database-0.7.0/images/sha256-2a4a724b7919a1ca362e89ca1a7dbb6e8201536386631a49fe8c69b4ebbf221c?context=explore) `docker pull apache/submarine:database-0.7.0`
    - [submarine jupyter-notebook](https://hub.docker.com/layers/submarine/apache/submarine/jupyter-notebook-0.7.0/images/sha256-0cacc189c7d2f220c23a89e6c9f0a542c274985f3a349e71613b5a92a0afea31?context=explore) `docker pull apache/submarine:jupyter-notebook-0.7.0`
    - [submarine quickstart](https://hub.docker.com/layers/submarine/apache/submarine/quickstart-0.7.0/images/sha256-eefbfde93d279a5bb69aecd74111addbdee4a5462eb0adb1805a0116532e75cb?context=explore) `docker pull apache/submarine:quickstart-0.7.0`
    - [submarine serve](https://hub.docker.com/layers/submarine/apache/submarine/serve-0.7.0/images/sha256-0bfed0744174c8c1d87fe8441f9fe006ab060ffcc2b207b4d013eef45267d103?context=explore) `docker pull apache/submarine:serve-0.7.0`
    - [submarine mlflow](https://hub.docker.com/layers/apache/submarine/mlflow-0.6.0/images/sha256-b395838b6c30e21c48c3304f20315788e2416bb4cf410779ad2d1530688e7fa9?context=explore) `docker pull apache/submarine:mlflow-0.7.0`
    - [submarine operator](https://hub.docker.com/layers/submarine/apache/submarine/operator-0.7.0/images/sha256-cd8b9a3c1e4a367ecf9df45e4ea8e78b9be0d347db5a70b3910cca87e73c4f28?context=explore) `docker pull apache/submarine:operator-0.7.0`
    - [submarine agent](https://hub.docker.com/layers/submarine/apache/submarine/agent-0.7.0/images/sha256-9c14c62478786eb9d7bbe74ca1aed48cd6ae4cb318bd9da149456926cd5c6474?context=explore) `docker pull apache/submarine:agent-0.7.0`

  - SDK:
    - [PySubmarine](https://pypi.org/project/apache-submarine/0.7.0/) `pip install apache-submarine==0.7.0`

- Apache Submarine `0.6.0` 于2021年10月21日发布 ([发布公告](/zh-cn/releases/submarine-release-0.6.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.6.0))
  - 二进制部署包:
    [submarine-dist-0.6.0-hadoop-2.9.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.6.0/submarine-dist-0.6.0-hadoop-2.9.tar.gz) (518 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.6.0/submarine-dist-0.6.0-hadoop-2.9.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.6.0/submarine-dist-0.6.0-hadoop-2.9.tar.gz.asc))
  - 源代码:
    [submarine-dist-0.6.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.6.0/submarine-dist-0.6.0-src.tar.gz) (8.3 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.6.0/submarine-dist-0.6.0-src.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.6.0/submarine-dist-0.6.0-src.tar.gz.asc)))
  - Docker 镜像:
    - [mini-submarine](https://hub.docker.com/layers/apache/submarine/mini-0.6.0/images/sha256-a068563409735c4e5c64d529936de614919b7fb9f11cc55c0302a19fe20bf37d?context=explore) `docker pull apache/submarine:mini-0.6.0`
    - [submarine server](https://hub.docker.com/layers/apache/submarine/server-0.6.0/images/sha256-e224668d76b7c758f67fdbfb1d478e26dfc49837eb49592da16041fe1ee1df2a?context=explore) `docker pull apache/submarine:server-0.6.0`
    - [submarine database](https://hub.docker.com/layers/apache/submarine/database-0.6.0/images/sha256-543bb90bc1c1dc6282934dbbaaae145f38fc494e134c916a17c49b69f171c911?context=explore) `docker pull apache/submarine:database-0.6.0`
    - [submarine jupyter-notebook](https://hub.docker.com/layers/apache/submarine/jupyter-notebook-0.6.0/images/sha256-c3464987598c2aee312f2e538b250dc2ec9d4b0ea15b760c67c52a7489e36130?context=explore) `docker pull apache/submarine:jupyter-notebook-0.6.0`
    - [submarine quickstart](https://hub.docker.com/layers/apache/submarine/quickstart-0.6.0/images/sha256-7f019c7fe71bbd34b5abced68736758908cc6f32696cf2c2a5f7b0d7200fde29?context=explore) `docker pull apache/submarine:quickstart-0.6.0`
    - [submarine serve](https://hub.docker.com/layers/apache/submarine/serve-0.6.0/images/sha256-d510a8e294a26b0c2f3043531dfd92b698adec1993f47171630ccc5612fe9930?context=explore) `docker pull apache/submarine:serve-0.6.0`
    - [submarine mlflow](https://hub.docker.com/layers/apache/submarine/mlflow-0.6.0/images/sha256-b395838b6c30e21c48c3304f20315788e2416bb4cf410779ad2d1530688e7fa9?context=explore) `docker pull apache/submarine:mlflow-0.6.0`
    - [submarine operator](https://hub.docker.com/layers/apache/submarine/operator-0.6.0/images/sha256-c7e7a0c47a9ddf693bbe01b28c707ac1f05a710a4b86e8baaf59395da13a9a42?context=explore) `docker pull apache/submarine:operator-0.6.0`

  - SDK:
    - [PySubmarine](https://pypi.org/project/apache-submarine/0.6.0/) `pip install apache-submarine==0.6.0`

- Apache Submarine `0.5.0` 于2020年12月17日发布 ([发布公告](/zh-cn/releases/submarine-release-0.5.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.5.0))
  - 二进制部署包:
    [submarine-dist-0.5.0-hadoop-2.9.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.5.0/submarine-dist-0.5.0-hadoop-2.9.tar.gz) (505 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-hadoop-2.9.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-hadoop-2.9.tar.gz.asc))
  - 源代码:
    [submarine-dist-0.5.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.5.0/submarine-dist-0.5.0-src.tar.gz) (5.0 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-src.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-src.tar.gz.asc)))
  - Docker 镜像:
    - [mini-submarine](https://hub.docker.com/layers/apache/submarine/mini-0.5.0/images/sha256-e3248c8c6336b245539028043783b91135eaffe9302dec05fe13571a0f2902a6) `docker pull apache/submarine:mini-0.5.0`
    - [submarine server](https://hub.docker.com/layers/apache/submarine/server-0.5.0/images/sha256-1805df8fd8e5274d16be8cdf39900d8576119c0caac7598db29990ebe138bf5c) `docker pull apache/submarine:server-0.5.0`
    - [submarine database](https://hub.docker.com/layers/apache/submarine/database-0.5.0/images/sha256-073889e773c1b44cef9f518dc2fc468ebc420200f6087e2a943438677dadc9e5) `docker pull apache/submarine:database-0.5.0`
    - [submarine jupyter-notebook](https://hub.docker.com/layers/apache/submarine/jupyter-notebook-0.5.0/images/sha256-f3cc2510c208b752ef4be7b383ee8f2325e4fc538696078bdb604d62fa47e4be) `docker pull apache/submarine:jupyter-notebook-0.5.0`
  - SDK:
    - [PySubmarine](https://pypi.org/project/apache-submarine/0.5.0/) `pip install apache-submarine==0.5.0`

- Apache Submarine `0.4.0`于2020年7月5日发布 ([发布公告](/zh-cn/releases/submarine-release-0.4.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.4.0))

  - 二进制部署包:
    [submarine-dist-0.4.0-hadoop-2.9.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.4.0/submarine-dist-0.4.0-hadoop-2.9.tar.gz) (550 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-hadoop-2.9.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-hadoop-2.9.tar.gz.asc))
  - 源代码:
    [submarine-dist-0.4.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.4.0/submarine-dist-0.4.0-src.tar.gz) (6 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-src.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-src.tar.gz.asc))
  - Docker 镜像:
    _[mini-submarine](https://hub.docker.com/layers/apache/submarine/mini-0.4.0/images/sha256-a8e7bd98f1f0325223d68e0ba64fd48bd56ee91736461d289945e70ad138e08f)_ [(guide)](https://github.com/apache/submarine/blob/rel/release-0.4.0/dev-support/mini-submarine/README.md#mini-submarine)

- Apache Submarine `0.3.0` 于2020年2月1日发布 ([发布公告](/zh-cn/releases/submarine-release-0.3.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.3.0))

  - submarine 二进制部署包:
    [submarine-dist-0.3.0-hadoop-2.9.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.3.0/submarine-dist-0.3.0-hadoop-2.9.tar.gz) (550 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-hadoop-2.9.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-hadoop-2.9.tar.gz.asc))
  - 源代码:
    [submarine-dist-0.3.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.3.0/submarine-dist-0.3.0-src.tar.gz) (6 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-src.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-src.tar.gz.asc))
  - Docker 镜像:
    _[mini-submarine](https://hub.docker.com/layers/apache/submarine/mini-0.3.0/images/sha256-3dd49054bf8a91521f5743c675278d626a5fa568e91651c67867b8ba6ceba340)_ [(guide)](https://github.com/apache/submarine/blob/rel/release-0.3.0/dev-support/mini-submarine/README.md#mini-submarine)

- Apache Submarine `0.2.0` 于2019年7月2日发布

  - submarine 二进制部署包:
    [hadoop-submarine-0.2.0.tar.gz](https://www.apache.org/dyn/closer.cgi/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0.tar.gz) (111 MB,
    [checksum](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0.tar.gz.mds),
    [signature](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0.tar.gz.asc),
    [Announcement](http://hadoop.apache.org/submarine/release/0.2.0/))

  - 源代码:
    [hadoop-submarine-0.2.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0-src.tar.gz) (1.4 MB,
    [checksum](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0-src.tar.gz.mds),
    [signature](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0-src.tar.gz.asc))

- Apache Submarine `0.1.0` 于2019年1月16日发布

  - submarine 二进制部署包:
    [submarine-0.2.0-bin-all.tgz](https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz) (97 MB,
    [checksum](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz.mds),
    [signature](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz.asc),
    [Announcement](https://hadoop.apache.org/docs/r3.2.0/index.html))

  - 源代码:
    [submarine-hadoop-3.2.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.2.0/hadoop-3.2.0-src.tar.gz) (1.1 MB,
    [checksum](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0-src.tar.gz.mds),
    [signature](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0-src.tar.gz.asc))
