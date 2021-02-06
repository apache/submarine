---
title: Guide for Apache Submarine Committers
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

This page contains Hadoop Core-specific guidelines for committers.

## New committers
New committers are encouraged to first read Apache's generic committer documentation:

* [Apache New Committer Guide](http://www.apache.org/dev/new-committers-guide.html)
* [Apache Committer FAQ](http://www.apache.org/dev/committers.html)

The first act of a new core committer is typically to add their name to the
credits page. This requires changing the site source in
https://github.com/apache/submarine-site/blob/master/community/member.md. Once done,
update the Submarine website as described
[here](https://github.com/apache/submarine-site/blob/asf-site/README.md)
(TLDR; don't forget to regenerate the site with hugo, and commit the generated
results, too).

## Review
Submarine committers should, as often as possible, attempt to review patches
submitted by others. Ideally every submitted patch will get reviewed by a
committer within a few days. If a committer reviews a patch they've not
authored, and believe it to be of sufficient quality, then they can commit the
patch, otherwise the patch should be cancelled with a clear explanation for why
it was rejected.

The list of submitted patches can be found in the GitHub
[Pull Requests](https://github.com/apache/submarine/pulls) page.
Committers should scan the list from top-to-bottom,
looking for patches that they feel qualified to review and possibly commit.

For non-trivial changes, it is best to get another committer to review & approve
your own patches before commit.

## Reject
Patches should be rejected which do not adhere to the guidelines in
[Contribution Guidelines](contributing.md). Committers should always be
polite to contributors and try to instruct and encourage them to contribute
better patches. If a committer wishes to improve an unacceptable patch, then it
should first be rejected, and a new patch should be attached by the committer
for review.

## Commit individual patches
Submarine uses git for source code version control. The writable repo is at -
https://gitbox.apache.org/repos/asf/submarine.git

It is strongly recommended to use the cicd script to merge the PRs.
See the instructions at
https://github.com/apache/submarine/tree/master/dev-support/cicd

## Adding Contributors role
There are three roles (Administrators, Committers, Contributors) in the project.

* Contributors who have Contributors role can become assignee of the issues in the project.
* Committers who have Committers role can set arbitrary roles in addition to Contributors role.
* Committers who have Administrators role can edit or delete all comments, or even delete issues in addition to Committers role.

How to set roles

1. Login to ASF JIRA
2. Go to the project page (e.g. https://issues.apache.org/jira/browse/SUBMARINE )
3. Hit "Administration" tab
4. Hit "Roles" tab in left side
5. Add Administrators/Committers/Contributors role
