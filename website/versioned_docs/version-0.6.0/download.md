---
title: Download Apache Submarine
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

The latest release of Apache Submarine is `0.5.0`.

  - Apache Submarine `0.5.0` released on Dec 17, 2020 ([release notes](https://submarine.apache.org/releases/submarine-release-0.5.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.5.0))
    * Binary package:
      [submarine-dist-0.5.0-hadoop-2.9.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.5.0/submarine-dist-0.5.0-hadoop-2.9.tar.gz) (505 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-hadoop-2.9.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-hadoop-2.9.tar.gz.asc))
    * Source:
      [submarine-dist-0.5.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.5.0/submarine-dist-0.5.0-src.tar.gz) (5.0 MB, [checksum](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-src.tar.gz.sha512), [signature](https://www.apache.org/dist/submarine/submarine-0.5.0/submarine-dist-0.5.0-src.tar.gz.asc)))
    * Docker images:
      * [mini-submarine](https://hub.docker.com/layers/apache/submarine/mini-0.5.0/images/sha256-e3248c8c6336b245539028043783b91135eaffe9302dec05fe13571a0f2902a6) `docker pull apache/submarine:mini-0.5.0`
      * [submarine server](https://hub.docker.com/layers/apache/submarine/server-0.5.0/images/sha256-1805df8fd8e5274d16be8cdf39900d8576119c0caac7598db29990ebe138bf5c) `docker pull apache/submarine:server-0.5.0`
      * [submarine database](https://hub.docker.com/layers/apache/submarine/database-0.5.0/images/sha256-073889e773c1b44cef9f518dc2fc468ebc420200f6087e2a943438677dadc9e5) `docker pull apache/submarine:database-0.5.0`
      * [submarine jupyter-notebook](https://hub.docker.com/layers/apache/submarine/jupyter-notebook-0.5.0/images/sha256-f3cc2510c208b752ef4be7b383ee8f2325e4fc538696078bdb604d62fa47e4be) `docker pull apache/submarine:jupyter-notebook-0.5.0`
    * SDK:
      * [PySubmarine](https://pypi.org/project/apache-submarine/0.5.0/) `pip install apache-submarine==0.5.0`

## Verify the integrity of the files

It is essential that you [verify](https://www.apache.org/info/verification.html) the integrity of the downloaded files using the PGP or MD5 signatures. This signature should be matched against the [KEYS](https://www.apache.org/dist/submarine/KEYS) file.

```
gpg --import KEYS
gpg --verify submarine-dist-X.Y.Z-src.tar.gz.asc
```

## Old releases
  - Apache Submarine 0.4.0 released on Jul 05, 2020 ([release notes](https://submarine.apache.org/releases/submarine-release-0.4.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.4.0))

    * Binary package with submarine:
    [submarine-dist-0.4.0-hadoop-2.9.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.4.0/submarine-dist-0.4.0-hadoop-2.9.tar.gz) (550 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-hadoop-2.9.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-hadoop-2.9.tar.gz.asc))
    * Source:
    [submarine-dist-0.4.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.4.0/submarine-dist-0.4.0-src.tar.gz) (6 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-src.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.4.0/submarine-dist-0.4.0-src.tar.gz.asc))
    * Docker images:
    *[mini-submarine](https://hub.docker.com/layers/apache/submarine/mini-0.4.0/images/sha256-a8e7bd98f1f0325223d68e0ba64fd48bd56ee91736461d289945e70ad138e08f)* [(guide)](https://github.com/apache/submarine/blob/rel/release-0.4.0/dev-support/mini-submarine/README.md#mini-submarine)
    
  - Apache Submarine 0.3.0 released on Feb 01, 2020 ([release notes](https://submarine.apache.org/releases/submarine-release-0.3.0)) ([git tag](https://github.com/apache/submarine/tree/rel/release-0.3.0))

    * Binary package with submarine:
    [submarine-dist-0.3.0-hadoop-2.9.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.3.0/submarine-dist-0.3.0-hadoop-2.9.tar.gz) (550 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-hadoop-2.9.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-hadoop-2.9.tar.gz.asc))
    * Source:
    [submarine-dist-0.3.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/submarine/submarine-0.3.0/submarine-dist-0.3.0-src.tar.gz) (6 MB,
    [checksum](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-src.tar.gz.sha512),
    [signature](https://www.apache.org/dist/submarine/submarine-0.3.0/submarine-dist-0.3.0-src.tar.gz.asc))
    * Docker images:
    *[mini-submarine](https://hub.docker.com/layers/apache/submarine/mini-0.3.0/images/sha256-3dd49054bf8a91521f5743c675278d626a5fa568e91651c67867b8ba6ceba340)* [(guide)](https://github.com/apache/submarine/blob/rel/release-0.3.0/dev-support/mini-submarine/README.md#mini-submarine)

  - Apache Submarine 0.2.0 released on Jul 2, 2019

    * Binary package with submarine:
    [hadoop-submarine-0.2.0.tar.gz](https://www.apache.org/dyn/closer.cgi/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0.tar.gz) (111 MB,
    [checksum](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0.tar.gz.mds),
    [signature](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0.tar.gz.asc),
    [Announcement](http://hadoop.apache.org/submarine/release/0.2.0/))

    * Source:
    [hadoop-submarine-0.2.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0-src.tar.gz) (1.4 MB,
    [checksum](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0-src.tar.gz.mds),
    [signature](https://dist.apache.org/repos/dist/release/hadoop/submarine/submarine-0.2.0/hadoop-submarine-0.2.0-src.tar.gz.asc))


  - Apache Submarine 0.1.0 released on Jan 16, 2019

    * Binary package with submarine:
    [submarine-0.2.0-bin-all.tgz](https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz) (97 MB,
    [checksum](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz.mds),
    [signature](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0.tar.gz.asc),
    [Announcement](https://hadoop.apache.org/docs/r3.2.0/index.html))

    * Source:
    [submarine-hadoop-3.2.0-src.tar.gz](https://www.apache.org/dyn/closer.cgi/hadoop/common/hadoop-3.2.0/hadoop-3.2.0-src.tar.gz) (1.1 MB,
    [checksum](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0-src.tar.gz.mds),
    [signature](https://www.apache.org/dist/hadoop/common/hadoop-3.2.0/hadoop-3.2.0-src.tar.gz.asc))
