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

Please refer to the following link to find more details about release guidelines:

[How to Release](https://cwiki.apache.org/confluence/display/SUBMARINE/How+to+release)  
[Submarine Release Guidelines](https://cwiki.apache.org/confluence/display/SUBMARINE/Submarine+Release+Guidelines)

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

Through the following URL, use the email to check whether the upload is successful or not. It will take about a minute to find out. When searching, check the show full-key hashes under advance on http://pgpkeys.mit.edu .

The query results are as follows:

### 1.5 Add your gpg public key to the KEYS file

> SVN is required for this step

The svn library of the DEV branch is https://dist.apache.org/repos/dist/dev/submarine

The SVN library of the Release branch is https://dist.apache.org/repos/dist/release/submarine

#### 1.5.1 Add the public key to KEYS in the dev branch to release the RC version

```shell
➜  ~ svn co https://dist.apache.org/repos/dist/dev/submarine /tmp/submarine-dist-dev
# This step is relatively slow, and all versions will be copied. If the network is disconnected, use svn cleanup to delete the lock and re-execute it, and the transfer will be resumed.
➜  ~ cd submarine-dist-dev
➜  submarine-dist-dev ~ (gpg --list-sigs YOUR_NAME@apache.org && gpg --export --armor YOUR_NAME@apache.org) >> KEYS # Append the KEY you generated to the file KEYS, it is best to check if it is correct after appending.
➜  submarine-dist-dev ~ svn add .	# If there is a KEYS file before, it is not needed.
➜  submarine-dist-dev ~ svn ci -m "add gpg key for YOUR_NAME" # Next, you will be asked to enter a username and password, just use your apache username and password.
```

#### 1.5.2 Add the public key to KEYS in the release branch to release the official version

```shell
➜  ~ svn co https://dist.apache.org/repos/dist/release/submarine /tmp/submarine-dist-release
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
$ git_tag=release-${release_version}-${rc_version}
$ git tag -s $git_tag -m "Tagging the ${release_version} first Releae Candidate (Candidates start at zero)"
# If a error happened like gpg: signing failed: secret key not available, set the private key first.
# Note: if you use zsh, you need to run the following command first: export GPG_TTY=$(tty)
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
cp ./submarine-dist/target/submarine-dist-${release_version}-${rc_version}.tar.gz /tmp/apache-submarine-${release_version}-${rc_version}/ # Copy the binary package to the source package directory to facilitate signing the package in the next step.
```

### 3.5 Sign the source package/binary package/sha512

```shell
for i in *.tar.gz; do echo $i; gpg --print-md SHA512 $i > $i.sha512 ; done # Calculate SHA512
for i in *.tar.gz; do echo $i; gpg --armor --output $i.asc --detach-sig $i ; done # Calculate the signature
```

### 3.6 Check whether the generated signature/sha512 is correct

<!-- For details, please refer to:[Verify](HowToVerify.md) -->

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
# Push to a remote repository, e.g.: https://gitbox.apache.org/repos/asf/submarine.git or https://github.com/apache/submarine.git
git push ${YOUR_REMOTE_NAME} ${git_tag}
```

### 4.3 Upload the compiled file to dist

> This step requires the use of SVN, the svn library of the DEV branch is https://dist.apache.org/repos/dist/dev/submarine

### 4.3.1 Checkout Submarine to a local directory

```shell
# This step may be slow, and all versions will be tested. If the network is broken, use svn cleanup to delete the lock and re-execute it, and the upload will be resumed.
svn co https://dist.apache.org/repos/dist/dev/submarine /tmp/submarine-dist-dev
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

Alternatively, you can create apache staging repository directly with one command as follows:
```shell
export GPG_PASSPHRASE=yourPassphase
export ASF_USERID=yourApacheId
export ASF_PASSWORD=yourApachePwd
cd $SUBMARINE_HOME/dev-support/cicd
./publish_release.sh ${release_version}-${rc_version} ${git_tag}
```

## 5. Enter voting

> To vote in the Submarine community, send an email to:`dev@submarine.apache.org`

### Vote in the Submarine community

#### Voting template

```
Title：[VOTE] Submarine-${release_version}-${rc_version} is ready for a vote!

Content：

Hi folks,

Thanks to everyone's help on this release.

I've created a release candidate (${rc_version}) for submarine ${release_version}. The
highlighted features are as follows:

1. AAA
2. BBB
3. CCC

The mini-submarine image is here:

docker pull apache/submarine:mini-${release_version}-${rc_version}


The RC tag in git is here:

https://github.com/apache/submarine/releases/tag/release-${release_version}-${rc_version}

The RC release artifacts are available at:

http://home.apache.org/~pingsutw/submarine-${release_version}-${rc_version}


The Maven staging repository is here:

https://repository.apache.org/content/repositories/orgapachesubmarine-1030

My public key is here:

https://dist.apache.org/repos/dist/release/submarine/KEYS


*This vote will run for 7 days, ending on DDDD/EE/FF at 11:59 pm PST.*


For the testing, I have verified the

1. Build from source, Install Submarine on minikube

2. Workbench UI (Experiment / Notebook / Template / Environment)

3. Experiment / Notebook / Template / Environment REST API


My +1 to start. Thanks!

BR,
XXX

```

#### Announce voting results template

```
Title：[RESULT][VOTE] Release Apache Submarine ${release_version} ${rc_version}

Content：

Hello Apache Submarine PMC and Community,
  The vote closes now as 72hr have passed. The vote PASSES with
  xx (+1 non-binding) votes from the PMC,
  xx (+1 non-binding) vote from the rest of the developer community,
  and no further 0 or -1 votes.

  The vote thread:{vote_mail_address}

Thank you for your support.
Your Submarine Release Manager
```

## 6. Officially released

### 6.1 Update release candidate version (like 0.8.0-RC0) to release version (like 0.8.0) in files from the release branch

In the meantime, we also need to change the version from 0.x.x.dev to 0.x.x in `submarine-sdk/pysubmarine/setup.py`.

### 6.2 Release the jira version

Access [submarine project version page](https://issues.apache.org/jira/projects/SUBMARINE?selectedItem=com.atlassian.jira.jira-projects-plugin:release-page&status=unreleased). Click the version to be released, and then click the "Release" button. It will request the release date. We can fill it with the end-of-vote date.

### 6.3 Tag the release
```shell
# please replace the version to the right version
export git_tag=rel/release-$release_version
git tag -s $git_tag -m "Submarine ${release_version} release"
git push ${YOUR_REMOTE_NAME} $git_tag
```

### 6.4 Package the source code and binary package

```shell
mkdir /tmp/apache-submarine-${release_version}
git archive --format=tar.gz --output="/tmp/apache-submarine-${release_version}/apache-submarine-${release_version}-src.tar.gz" --prefix="apache-submarine-${release_version}/" $git_tag

cd /tmp/apache-submarine-${release_version} 
tar xzvf apache-submarine-${release_version}-src.tar.gz 
cd apache-submarine-${release_version} 
mvn compile clean install package -DskipTests 
cp ./submarine-dist/target/submarine-dist-${release_version}.tar.gz /tmp/apache-submarine-${release_version}/  
```

### 6.5 Sign and check the source package/binary package/sha512

Sign:

```shell
for i in *.tar.gz; do echo $i; gpg --print-md SHA512 $i > $i.sha512 ; done 
for i in *.tar.gz; do echo $i; gpg --armor --output $i.asc --detach-sig $i ; done 
```

Check:

```shell
for i in *.tar.gz; do echo $i; gpg --verify $i.asc $i ; done
```

### 6.6 Copy release artifacts to apache dist server

```shell
svn co --depth immediates https://dist.apache.org/repos/dist /tmp/submarine-dist-release
cd /tmp/submarine-dist-release
svn update --set-depth immediates dev/submarine

# upload to dev
cd dev/submarine
mkdir $release_version
cp /tmp/apache-submarine-${release_version}/*tar.gz* ${release_version}/
svn add $release_version
svn ci -m "Publishing the bits for submarine release ${release_version} to apache dist dev folder"

# upload to release
cd /tmp/submarine-dist-release
svn update --set-depth immediates release/submarine
svn copy dev/submarine/$release_version release/submarine/
svn ci -m "Publishing the bits for submarine release ${release_version}"
```

### 6.7 Release the version in the Apache Staging repository

You need to execute the deploy command again.
```shell
mvn -DskipTests deploy -Papache-release -Dmaven.javadoc.skip=true
```

> Please make sure all artifacts are fine.

1. Log in to http://repository.apache.org with your Apache account.
2. Click on Staging repositories on the left.
3. Search for Submarine keywords, select your recently uploaded repository, the repository specified in the voting email.
4. Click the `Release` button above, and a series of checks will be carried out during this process.
   **It usually takes 24 hours to wait for the repository to synchronize to other data sources**

### 6.8 Release Python SDK

More details can be found in https://github.com/apache/submarine/blob/master/website/docs/userDocs/submarine-sdk/pysubmarine/development.md#upload-package-to-pypi .

### 6.9 Update official website link

Create a new folder `website/versioned_docs/version-${release_version}`, and copy the files under `website/docs/*` to `website/versioned_docs/version-${release_version}`.

This may need a new PR and be committed to the master branch.

### 6.10 Update doap_Submarine.rdf

Update the DOAP file with the release version and release date.

```xml
<release>
  <Version>
    <name>Apache Submarine x.y.z</name>
    <created>YYYY-MM-DD</created>
    <revision>x.y.z</revision>
  </Version>
</release>
```

### 6.11 Send an email to`dev@submarine.apache.org`

**Please make sure that the repository in 6.4 has been successfully released, generally the email is sent 24 hours after 6.4**

Announce release email template:

```
Title： [ANNOUNCE] Apache Submarine ${release_version} release!
Content：
Hi folks, It's a great honor for me to announce that the Apache Submarine Community
has released Apache Submarine ${release_version}!
The highlighted features are:
1. AAA
2. BBB
3. CCC

Tons of thanks to our contributors and community!
Let's keep fighting! *Apache Submarine ${release_version} released*:
https://submarine.apache.org/releases/submarine-release-${release_version}

BR,
XXXX
```
