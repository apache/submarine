---
title: How to Release
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

> This article mainly introduces how the Release Manager releases a specific version of the project according to the Apache process.

## 0. Preface

Source Release is the focus of Apache’s attention and it is also a required content for release. Binary Release is optional, Submarine can choose whether to release the binary package to the Apache warehouse or to the Maven central warehouse.

Please refer to the following link to find more ASF release guidelines:

[Apache Release Guide](https://incubator.apache.org/guides/releasemanagement.html)

## 1. Add GPG KEY

> Main references in this chapter:https://infra.apache.org/openpgp.html > **This chapter is only needed for the first release manager of the project.**

### 1.1 Install gpg

Detailed installation documents can refer to [tutorial](https://www.gnupg.org/download/index.html), The environment configuration of Mac OS is as follows:

```shell
$ brew install gpg
$ gpg --version #Check the version，should be 2.x
```

### 1.2 generate gpg Key

#### Need to pay attention to the following points：

- When entering the name, it is better to be consistent with the Full name registered in Apache
- The mailbox used should be apache mailbox
- It’s better to use pinyin or English for the name, otherwise there will be garbled characters

#### Follow the hint，generate a key

```shell
➜  ~ gpg --full-gen-key
gpg (GnuPG) 2.2.20; Copyright (C) 2020 Free Software Foundation, Inc.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Please select what kind of key you want:
   (1) RSA and RSA (default)
   (2) DSA and Elgamal
   (3) DSA (sign only)
   (4) RSA (sign only)
  (14) Existing key from card
Your selection? 1 # enter 1 here
RSA keys may be between 1024 and 4096 bits long.
What keysize do you want? (2048) 4096 # enter 4096 here
Requested keysize is 4096 bits
Please specify how long the key should be valid.
         0 = key does not expire
      <n>  = key expires in n days
      <n>w = key expires in n weeks
      <n>m = key expires in n months
      <n>y = key expires in n years
Key is valid for? (0) 0 # enter 0 here
Key does not expire at all
Is this correct? (y/N) y # enter y here

GnuPG needs to construct a user ID to identify your key.

Real name: Guangxu Cheng # enter your name here
Email address: gxcheng@apache.org # enter your mailbox here
Comment:                          # enter some comment here (Optional)
You selected this USER-ID:
    "Guangxu Cheng <gxcheng@apache.org>"

Change (N)ame, (C)omment, (E)mail or (O)kay/(Q)uit? O #enter O here
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.
We need to generate a lot of random bytes. It is a good idea to perform
some other action (type on the keyboard, move the mouse, utilize the
disks) during the prime generation; this gives the random number
generator a better chance to gain enough entropy.

# A dialog box will pop up, asking you to enter the key for this gpg.
┌──────────────────────────────────────────────────────┐
│ Please enter this passphrase                         │
│                                                      │
│ Passphrase: _______________________________          │
│                                                      │
│       <OK>                              <Cancel>     │
└──────────────────────────────────────────────────────┘
# After entering the secret key, it will be created. And it will output the following information.
gpg: key 2DD587E7B10F3B1F marked as ultimately trusted
gpg: revocation certificate stored as '/Users/cheng/.gnupg/openpgp-revocs.d/41936314E25F402D5F7D73152DD587E7B10F3B1F.rev'
public and secret key created and signed.

pub   rsa4096 2020-05-19 [SC]
      41936314E25F402D5F7D73152DD587E7B10F3B1F
uid                      Guangxu Cheng <gxcheng@apache.org>
sub   rsa4096 2020-05-19 [E]
```

### 1.3 Upload the generated key to the public server

```shell
➜  ~ gpg --list-keys
-------------------------------
pub   rsa4096 2020-05-18 [SC]
      5931F8CFD04B37A325E4465D8C0D31C4149B3A87
uid           [ultimate] Guangxu Cheng <gxcheng@apache.org>
sub   rsa4096 2020-05-18 [E]

# Send public key to keyserver via key id
$ gpg --keyserver pgpkeys.mit.edu --send-key <key id>
# Among them, pgpkeys.mit.edu is a randomly selected keyserver, and the keyserver list is: https://sks-keyservers.net/status/, which is automatically synchronized with each other, you can choose any one.
```

### 1.4 Check whether the key is created successfully

Through the following URL, use the email to check whether the upload is successful or not. It will take about a minute to find out. When searching, check the show full-key hashes under advance on http://keys.gnupg.net.

The query results are as follows:

### 1.5 Add your gpg public key to the KEYS file

> SVN is required for this step

The svn library of the DEV branch is https://dist.apache.org/repos/dist/dev/incubator/Submarine

The SVN library of the Release branch is https://dist.apache.org/repos/dist/release/incubator/submarine

#### 1.5.1 Add the public key to KEYS in the dev branch to release the RC version

```shell
➜  ~ svn co https://dist.apache.org/repos/dist/dev/incubator/submarine /tmp/submarine-dist-dev
# This step is relatively slow, and all versions will be copied. If the network is disconnected, use svn cleanup to delete the lock and re-execute it, and the transfer will be resumed.
➜  ~ cd submarine-dist-dev
➜  submarine-dist-dev ~ (gpg --list-sigs YOUR_NAME@apache.org && gpg --export --armor YOUR_NAME@apache.org) >> KEYS # Append the KEY you generated to the file KEYS, it is best to check if it is correct after appending.
➜  submarine-dist-dev ~ svn add .	# If there is a KEYS file before, it is not needed.
➜  submarine-dist-dev ~ svn ci -m "add gpg key for YOUR_NAME" # Next, you will be asked to enter a username and password, just use your apache username and password.
```

#### 1.5.2 Add the public key to KEYS in the release branch to release the official version

```shell
➜  ~ svn co https://dist.apache.org/repos/dist/release/incubator/submarine /tmp/submarine-dist-release
➜  ~ cd submarine-dist-release
➜  submarine-dist-release ~ (gpg --list-sigs YOUR_NAME@apache.org && gpg --export --armor YOUR_NAME@apache.org) >> KEYS	# Append the KEY you generated to the file KEYS, it is best to check if it is correct after appending.
➜  submarine-dist-release ~ svn add .	# If there is a KEYS file before, it is not needed.
➜  submarine-dist-release ~ svn ci -m "add gpg key for YOUR_NAME" # Next, you will be asked to enter a username and password, just use your apache username and password.
```

### 1.6 Upload GPG public key to Github account

1. Go to https://github.com/settings/keys and add GPG KEYS.
2. If you find "unverified" is written after the key after adding it, remember to bind the mailbox used in the GPG key to your github account (https://github.com/settings/emails).

## 2. Set maven settings

**Skip if it has already been set**

In the maven configuration file ~/.m2/settings.xml, add the following `<server>` item

```xml
<?xml version="1.0" encoding="UTF-8"?>
<settings xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.1.0 http://maven.apache.org/xsd/settings-1.1.0.xsd" xmlns="http://maven.apache.org/SETTINGS/1.1.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <servers>
    <!-- Apache Repo Settings -->
    <server>
        <id>apache.snapshots.https</id>
        <username>{user-id}</username>
        <password>{user-pass}</password>
    </server>
    <server>
        <id>apache.releases.https</id>
        <username>{user-id}</username>
        <password>{user-pass}</password>
    </server>
  </servers>
<profiles>
    <profile>
      <id>apache-release</id>
      <properties>
        <gpg.keyname>Your KEYID</gpg.keyname><!-- Your GPG Keyname here -->
        <!-- Use an agent: Prevents being asked for the password during the build -->
        <gpg.useagent>true</gpg.useagent>
        <gpg.passphrase>Your password of the private key</gpg.passphrase>
      </properties>
    </profile>
</profiles>
</settings>
```

## 3. Compile and package

### 3.1 Prepare a branch

- Pull the new branch from the main branch as a release branch, release-${release_version}

- Update `CHANGES.md`

- Check whether the code is normal, including successful compilation, all unit tests, successful RAT check, etc.

  ```shell
  # build check
  $ mvn clean package -Dmaven.javadoc.skip=true
  # RAT check
  $ mvn apache-rat:check
  ```

- Change the version number

### 3.2 Create the tag

> Before creating the tag, make sure that the code has been checked for errors, including: successful compilation, all unit tests, and successful RAT checks, etc.

**Create a tag with signature**

```shell
$ git_tag=${release_version}-${rc_version}
$ git tag -s $git_tag -m "Tagging the ${release_version} first Releae Candidate (Candidates start at zero)"
# If a error happened like gpg: signing failed: secret key not available, set the private key first.
$ git config user.signingkey ${KEY_ID}
```

### 3.3 Package the source code

> After the tag is successfully created, the tag source code should be packaged into a tar package.

```shell
mkdir /tmp/apache-submarine-${release_version}-${rc_version}
git archive --format=tar.gz --output="/tmp/apache-submarine-${release_version}-${rc_version}/apache-submarine-${release_version}-src.tar.gz" --prefix="apache-submarine-${release_version}/" $git_tag
```

### 3.4 Packaged binary package

> Compile the source code packaged in the previous step

```shell
cd /tmp/apache-submarine-${release_version}-${rc_version} # Enter the source package directory.
tar xzvf apache-submarine-${release_version}-src.tar.gz # Unzip the source package.
cd apache-submarine-${release_version} # Enter the source directory.
mvn compile clean install package -DskipTests # Compile.
cp ./submarine-distribution/target/apache-submarine-${release_version}-bin.tar.gz /tmp/apache-submarine-${release_version}-${rc_version}/  # Copy the binary package to the source package directory to facilitate signing the package in the next step.
```

### 3.5 Sign the source package/binary package/sha512

```shell
for i in *.tar.gz; do echo $i; gpg --print-md SHA512 $i > $i.sha512 ; done # Calculate SHA512
for i in *.tar.gz; do echo $i; gpg --armor --output $i.asc --detach-sig $i ; done # Calculate the signature
```

### 3.6 Check whether the generated signature/sha512 is correct

For details, please refer to:[Verify](how-to-verify.md)
For example, verify that the signature is correct as follows:

```shell
for i in *.tar.gz; do echo $i; gpg --verify $i.asc $i ; done
```

## 4. Prepare for Apache release

### 4.1 Publish the jar package to the Apache Nexus repository

```shell
cd /tmp/apache-submarine-${release_version}-${rc_version} # Enter the source package directory
tar xzvf apache-submarine-${release_version}-src.tar.gz # Unzip the source package
cd apache-submarine-${release_version}
mvn -DskipTests deploy -Papache-release -Dmaven.javadoc.skip=true  # Start upload
```

### 4.2 Upload the tag to git repository

```shell
git push origin ${release_version}-${rc_version}
```

### 4.3 Upload the compiled file to dist

> This step requires the use of SVN, the svn library of the DEV branch is https://dist.apache.org/repos/dist/dev/incubator/submarine

### 4.3.1 Checkout Submarine to a local directory

```shell
# This step may be slow, and all versions will be tested. If the network is broken, use svn cleanup to delete the lock and re-execute it, and the upload will be resumed.
svn co https://dist.apache.org/repos/dist/dev/incubator/submarine /tmp/submarine-dist-dev
```

### 4.3.2 Add the public key to the KEYS file and submit it to the SVN repository

```shell
cd /tmp/submarine-dist-dev
mkdir ${release_version}-${rc_version} # Create version directory
# Copy the source code package and signed package here.
cp /tmp/apache-submarine-${release_version}-${rc_version}/*tar.gz* ${release_version}-${rc_version}/
svn status # Check svn status.
svn add ${release_version}-${rc_version} # Add to svn version.
svn status # Check svn status.
svn commit -m "prepare for ${release_version} ${rc_version}"     # Submit to svn remote server.
```

### 4.4 Shut down the Apache Staging repository

> Please make sure all artifacts are fine.

1. **Log in** http://repository.apache.org , with Apache account
2. Click on Staging repositories on the left.
3. Search for Submarine keywords and select the repository you uploaded recently.
4. Click the Close button above, and a series of checks will be performed during this process.
5. After the check is passed, a link will appear on the Summary tab below. Please save this link and put it in the next voting email.
   The link should look like: `https://repository.apache.org/content/repositories/orgapachesubmarine-xxxx`

WARN: Please note that clicking Close may fail, please check the reason for the failure and deal with it.

## 5. Enter voting

> To vote in the Submarine community, send an email to:`dev@submarine.apache.org`

### Vote in the Submarine community

#### Voting template

```html
Title：[VOTE] Release Apache Submarine ${release_version} ${rc_version}
Content： Hello Apache Submarine PPMC and Community, This is a call for a vote
to release Apache Submarine version ${release_version}-${rc_version}. The tag to
be voted on is ${release_version}-${rc_version}:
https://github.com/apache/incubator-submarine/tree/${release_version}-${rc_version}
The release tarball, signature, and checksums can be found at:
https://dist.apache.org/repos/dist/dev/incubator/submarine/${release_version}-${rc_version}/
Maven artifacts are available in a staging repository at:
https://repository.apache.org/content/repositories/orgapachesubmarine-{staging-id}
Artifacts were signed with the {YOUR_PUB_KEY} key which can be found in:
https://downloads.apache.org/incubator/submarine/KEYS ${release_version}
includes ~ ${issue_count} bug fixes and improvements done since last versions
which can be found at:
https://github.com/apache/incubator-submarine/blob/${release_version}-${rc_version}/CHANGES.md
Please download, verify, and test. The VOTE will remain open for at least 72
hours. [ ] +1 Release this package as Apache Submarine ${release_version} [ ] +0
[ ] -1 Do not release this package because... To learn more about apache
Submarine, please see http://submarine.apache.org/ Checklist for reference: [ ]
Download links are valid. [ ] Checksums and signatures. [ ]
LICENSE/NOTICE/DISCLAIMER files exist [ ] No unexpected binary files [ ] All
source files have ASF headers [ ] Can compile from source [ ] All Tests Passed
More detailed checklist please refer to:
https://cwiki.apache.org/confluence/display/INCUBATOR/Incubator+Release+Checklist
Thanks, Your Submarine Release Manager
```

#### Announce voting results template

```html
Title：[RESULT][VOTE] Release Apache Submarine ${release_version} ${rc_version}
Content： Hello Apache Submarine PPMC and Community, The vote closes now as 72hr
have passed. The vote PASSES with xx (+1 non-binding) votes from the PPMC, xx
(+1 binding) vote from the IPMC, xx (+1 non-binding) vote from the rest of the
developer community, and no further 0 or -1 votes. The vote thread:
{vote_mail_address} I will now bring the vote to general@incubator.apache.org to
get approval by the IPMC. If this vote passes also, the release is accepted and
will be published. Thank you for your support. Your Submarine Release Manager
```

## 6. Officially released

### 6.1 Merge the changes from the release-${release_version} branch to the master branch

### 6.2 Release the version in the Apache Staging repository

> Please make sure all artifacts are fine.

1. Log in to http://repository.apache.org with your Apache account.
2. Click on Staging repositories on the left.
3. Search for Submarine keywords, select your recently uploaded repository, the repository specified in the voting email.
4. Click the `Release` button above, and a series of checks will be carried out during this process.
   **It usually takes 24 hours to wait for the repository to synchronize to other data sources**

### 6.3 Update official website link

### 6.4. Send an email to`dev@submarine.apache.org`

**Please make sure that the repository in 6.4 has been successfully released, generally the email is sent 24 hours after 6.4**

Announce release email template:

```html
Title： [ANNOUNCE] Release Apache Submarine(incubating) ${release_version}
Content： Hi all, The Apache Submarine(incubating) community is pleased to
announce that Apache Submarine(incubating) ${release_version} has been released!
Apache Submarine is a one-stop data streaming platform that provides automatic,
secure, distributed, and efficient data publishing and subscription
capabilities. This platform helps you easily build stream-based data
applications. Download Links: https://submarine.apache.org/download/main Release
Notes: https://submarine.apache.org/download/release-${release_version} Website:
https://submarine.apache.org/ Submarine Resources: - Issue:
https://github.com/apache/incubator-submarine/issues - Mailing list:
dev@submarine.apache.org Thanks On behalf of Apache Submarine(Incubating)
community
```
