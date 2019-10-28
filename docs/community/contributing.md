<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribution Guidelines

**Apache Hadoop Submarine** is an [Apache 2.0 License](https://github.com/apache/hadoop-submarine/blob/master/LICENSE) Software.

Contributing to Hadoop Submarine (Source code, Documents, Image, Website) means you agree to the Apache 2.0 License.

1. Make sure your issue is not already in the [Jira issue tracker](https://issues.apache.org/jira/browse/SUBMARINE)
2. If not, create a ticket describing the change you're proposing in the [Jira issue tracker](https://issues.apache.org/jira/browse/SUBMARINE)
3. Setup Travis [Continuous Integration](#continuous-integration)
4. Contribute your patch via Pull Request on our [Github Mirror](https://github.com/apache/hadoop-submarine).

Before you start, please read the [Code of Conduct](http://www.apache.org/foundation/policies/conduct.html) carefully, familiarize yourself with it and refer to it whenever you need it.

For those of you who are not familiar with the Apache project, understanding [How it works](http://www.apache.org/foundation/how-it-works.html) would be quite helpful.

## Creating a Pull Request
When creating a Pull Request, you will automatically get the template below.

Filling it thoroughly can improve the speed of the review process.

    ### What is this PR for?
    A few sentences describing the overall goals of the pull request's commits.
    First time? Check out the contribution guidelines - 
    https://github.com/apache/hadoop-submarine/tree/master/docs/community/contributing.md

    ### What type of PR is it?
    [Bug Fix | Improvement | Feature | Documentation | Hot Fix | Refactoring]

    ### Todos
    * [ ] - Task

    ### What is the Jira issue?
    * Open an issue on Jira https://issues.apache.org/jira/browse/SUBMARINE/
    * Put link here, and add [SUBMARINE-${jira_number}] in PR title, e.g. [SUBMARINE-323]

    ### How should this be tested?
    Outline the steps to test the PR here.

    ### Screenshots (if appropriate)

    ### Questions:
    * Do the licenses files require updates?
    * Are there breaking changes for older versions?
    * Does this need documentation?


## Source Control Workflow
Hadoop Submarine follows [Fork & Pull](https://github.com/sevntu-checkstyle/sevntu.checkstyle/wiki/Development-workflow-with-Git:-Fork,-Branching,-Commits,-and-Pull-Request) model.

## The Review Process

When a Pull Request is submitted, it is being merged or rejected by the following review process.

* Anybody can be a reviewer and may comment on the change or suggest modifications.
* Reviewer can indicate that a patch looks suitable for merging with a comment such as: "Looks good", "LGTM", "+1".
* At least one indication of suitability (e.g. "LGTM") from a committer is required to be merged.
* Pull request is open for 1 or 2 days for potential additional review unless it's got enough indication of suitability.
* A committer can then initiate lazy consensus ("Merge if there is no more discussion") after which the code can be merged after a particular time (usually 24 hours) if there are no more reviews.
* Contributors can ping reviewers (including committers) by commenting 'Ready to review' or suitable indication.


## Setting up
Here are some things you will need to build and test the Hadoop Submarine.

### Software Configuration Management (SCM)

Hadoop Submarine uses Git for its SCM system. So you'll need a git client installed on your development machine.

### Integrated Development Environment (IDE)

You are free to use whatever IDE you prefer, or your favorite command-line editor.

### Code convention
We are following Google Code style:

* [Java style](https://google.github.io/styleguide/javaguide.html)
* [Shell style](https://google.github.io/styleguide/shell.xml)

There are some plugins to format, lint your code in IDE (use [dev-support/maven-config/checkstyle.xml](hhttps://github.com/apache/hadoop-submarine/blob/master/dev-support/maven-config/checkstyle.xml) as rules)

* [Checkstyle plugin for Intellij](https://plugins.jetbrains.com/plugin/1065) ([Setting Guide](http://stackoverflow.com/questions/26955766/intellij-idea-checkstyle))
* [Checkstyle plugin for Eclipse](http://eclipse-cs.sourceforge.net/#!/) ([Setting Guide](http://eclipse-cs.sourceforge.net/#!/project-setup))


## Getting the source code

### Step 1: Fork in the cloud

1. Visit https://github.com/apache/hadoop-submarine
2. On the top right of the page, click the `Fork` button (top right) to create a cloud-based fork of the repository.

### Step 2: Clone fork to local storage

Create your clone:

> ${user} is your github user name

```sh
mkdir -p ${working_dir}
cd ${working_dir}

git clone https://github.com/${user}/hadoop-submarine.git
# or: git clone git@github.com:${user}/hadoop-submarine.git

cd ${working_dir}/hadoop-submarine
git remote add upstream https://github.com/apache/hadoop-submarine.git
# or: git remote add upstream git@github.com:apache/hadoop-submarine.git

# Never push to the upstream master.
git remote set-url --push upstream no_push

# Confirm that your remotes make sense:
# It should look like:
# origin    git@github.com:${user}/hadoop-submarine.git (fetch)
# origin    git@github.com:${user}/hadoop-submarine.git (push)
# upstream  https://github.com/apache/hadoop-submarine (fetch)
# upstream  no_push (push)
git remote -v
```

### Step 3: Branch

Get your local master up to date:

```sh
cd ${working_dir}/hadoop-submarine
git fetch upstream
git checkout master
git rebase upstream/master
```

Branch from master:

```sh
git checkout -b SUBMARINE-${jira_number}
```

### Step 4: Develop

#### Edit the code

You can now edit the code on the `SUBMARINE-${jira_number}` branch.

#### Test

Build and run all tests:

### Step 5: Keep your branch in sync

```sh
# While on your SUBMARINE-${jira_number} branch.
git fetch upstream
git rebase upstream/master
```

Please don't use `git pull` instead of the above `fetch`/`rebase`. `git pull` does a merge, which leaves merge commits. These make the commit history messy and violate the principle that commits ought to be individually understandable and useful (see below). You can also consider changing your `.git/config` file via `git config branch.autoSetupRebase` always to change the behavior of `git pull`.

### Step 6: Commit

Commit your changes.

```sh
git commit
```

Likely you'll go back and edit/build/test further and then `commit --amend` in a few cycles.

### Step 7: Push

When the changes are ready to review (or you just want to create an offsite backup of your work), push your branch to your fork on `github.com`:

```sh
git push --set-upstream ${your_remote_name} SUBMARINE-${jira_number}
```

### Step 8: Create a pull request

1. Visit your fork at `https://github.com/${user}/hadoop-submarine`.
2. Click the `Compare & Pull Request` button next to your `SUBMARINE-${jira_number}` branch.
3. Fill in the required information in the PR template.

#### Get a code review

If your pull request (PR) is opened, it will be assigned to one or more reviewers. Those reviewers will do a thorough code review, looking at correctness, bugs, opportunities for improvement, documentation comments, and style.

To address review comments, you should commit the changes to the same branch of the PR on your fork.

#### Revert a commit

In case you wish to revert a commit, follow the instructions below:

> NOTE: If you have upstream write access, please refrain from using the Revert
> button in the GitHub UI for creating the PR, because GitHub will create the
> PR branch inside the main repository rather than inside your fork.

Create a branch and synchronize it with the upstream:

```sh
# create a branch
git checkout -b myrevert

# sync the branch with upstream
git fetch upstream
git rebase upstream/master

# SHA is the hash of the commit you wish to revert
git revert SHA
```

This creates a new commit reverting the change. Push this new commit to your remote:

```sh
git push ${your_remote_name} myrevert
```

Create a PR based on this branch.

#### Cherry pick a commit to a release branch

In case you wish to cherry pick a commit to a release branch, follow the instructions below:

Create a branch and synchronize it with the upstream:

```sh
# sync the branch with upstream.
git fetch upstream

# checkout the release branch.
# ${release_branch_name} is the release branch you wish to cherry pick to.
git checkout upstream/${release_branch_name}
git checkout -b my-cherry-pick

# cherry pick the commit to my-cherry-pick branch.
# ${SHA} is the hash of the commit you wish to revert.
git cherry-pick ${SHA}

# push this branch to your repo, file an PR based on this branch.
git push --set-upstream ${your_remote_name} my-cherry-pick
```

## Build

[Build From Code](../development/BuildFromCode.md)

## Continuous Integration

Hadoop Submarine project's CI system will collect information from pull request author's Travis-ci and display status in the pull request.

Each individual contributor should setup Travis-ci for the fork before making a pull-request. Go to [https://travis-ci.org/profile](https://travis-ci.org/profile) and switch on 'hadoop-submarine' repository.