---
title: How to Verify
---

<!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->

### Verification of the release candidate

## 1. Download the candidate version to be released to the local environment

```shell
svn co https://dist.apache.org/repos/dist/dev/submarine/${release_version}-${rc_version}/
```

## 2. Verify whether the uploaded version is compliant

> Begin the verification process, which includes but is not limited to the following content and forms.

### 2.1 Check if the release package is complete

> The package uploaded to dist must include the source code package, and the binary package is optional.

1. Whether it includes the source code package.
2. Whether it includes the signature of the source code package.
3. Whether it includes the sha512 of the source code package.
4. If the binary package is uploaded, also check the contents listed in (2)-(4).

### 2.2 Check gpg signature

- Import the public key

```shell
curl https://dist.apache.org/repos/dist/dev/submarine/KEYS > KEYS # Download KEYS
gpg --import KEYS # Import KEYS to local
```

- Trust the public key
  > Trust the KEY used in this version.

```
  gpg --edit-key xxxxxxxxxx # The KEY used in this version
  gpg (GnuPG) 2.2.21; Copyright (C) 2020 Free Software Foundation, Inc.
  This is free software: you are free to change and redistribute it.
  There is NO WARRANTY, to the extent permitted by law.

  Secret key is available.

  sec  rsa4096/5EF3A66D57EC647A
       created: 2020-05-19  expires: never       usage: SC
       trust: ultimate      validity: ultimate
  ssb  rsa4096/17628566FEED6AF7
       created: 2020-05-19  expires: never       usage: E
  [ultimate] (1). XXX YYYZZZ <yourAccount@apache.org>

  gpg> trust
  sec  rsa4096/5EF3A66D57EC647A
       created: 2020-05-19  expires: never       usage: SC
       trust: ultimate      validity: ultimate
  ssb  rsa4096/17628566FEED6AF7
       created: 2020-05-19  expires: never       usage: E
  [ultimate] (1). XXX YYYZZZ <yourAccount@apache.org>

  Please decide how far you trust this user to correctly verify other users' keys
  (by looking at passports, checking fingerprints from different sources, etc.)

    1 = I don't know or won't say
    2 = I do NOT trust
    3 = I trust marginally
    4 = I trust fully
    5 = I trust ultimately
    m = back to the main menu

  Your decision? 5 #choose 5
  Do you really want to set this key to ultimate trust? (y/N) y # choose y

  sec  rsa4096/5EF3A66D57EC647A
       created: 2020-05-19  expires: never       usage: SC
       trust: ultimate      validity: ultimate
  ssb  rsa4096/17628566FEED6AF7
       created: 2020-05-19  expires: never       usage: E
  [ultimate] (1). XXX YYYZZZ <yourAccount@apache.org>

  gpg>

  sec  rsa4096/5EF3A66D57EC647A
       created: 2020-05-19  expires: never       usage: SC
       trust: ultimate      validity: ultimate
  ssb  rsa4096/17628566FEED6AF7
       created: 2020-05-19  expires: never       usage: E
  [ultimate] (1). XXX YYYZZZ <yourAccount@apache.org>
```

- Use the following command to check the signature.

```shell
for i in *.tar.gz; do echo $i; gpg --verify $i.asc $i ; done
#Or
gpg --verify apache-submarine-${release_version}-src.tar.gz.asc apache-submarine-${release_version}-src.tar.gz
# If you upload a binary package, you also need to check whether the signature of the binary package is correct.
gpg --verify apache-submarine-server-${release_version}-bin.tar.gz.asc apache-submarine-server-${release_version}-bin.tar.gz
gpg --verify apache-submarine-client-${release_version}-bin.tar.gz.asc apache-submarine-client-${release_version}-bin.tar.gz
```

- Check the result
  > If something like the following appears, it means that the signature is correct. The keyword：**`Good signature`**

```shell
apache-submarine-${release_version}-src.tar.gz
gpg: Signature made Sat May 30 11:45:01 2020 CST
gpg:                using RSA key 9B12C2228BDFF4F4CFE849445EF3A66D57EC647A
gpg: Good signature from "XXX YYYZZZ <yourAccount@apache.org>" [ultimate]gular2
```

### 2.3 Check sha512 hash

> After calculating the sha512 hash locally, verify whether it is consistent with the one on dist.

```shell
for i in *.tar.gz; do echo $i; gpg --print-md SHA512 $i; done
#Or
gpg --print-md SHA512 apache-submarine-${release_version}-src.tar.gz
# If you upload a binary package, you also need to check the sha512 hash of the binary package.
gpg --print-md SHA512 apache-submarine-server-${release_version}-bin.tar.gz
gpg --print-md SHA512 apache-submarine-client-${release_version}-bin.tar.gz
# 或者
for i in *.tar.gz.sha512; do echo $i; sha512sum -c $i; done
```

### 2.4. Check the file content of the source package.

Unzip `apache-submarine-${release_version}-src.tar.gz` and check as follows:

- Whether the DISCLAIMER file exists and whether the content is correct.
- Whether the LICENSE and NOTICE file exists and whether the content is correct.
- Whether all files have ASF License header.
- Whether the source code can be compiled normally.
- Whether the single test is passed.
- ....

### 2.5 Check the binary package (if the binary package is uploaded)

Unzip `apache-submarine-client-${release_version}-src.tar.gz` and ` apache-submarine-server-${release_version}-src.tar.gz`, then check as follows:

- Whether the DISCLAIMER file exists and whether the content is correct.
- Whether the LICENSE and the NOTICE file exists and whether the content is correct.
- Whether the deployment is successful.
- Deploy a test environment to verify whether production and consumption can run normally.
- Verify what you think might go wrong.
