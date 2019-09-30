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

# Terminology

| Term | Description |
| -------- | -------- |
| User | A single data-scientist/data-engineer. User has resource quota, credentials |
| Team | User belongs to one or more teams, teams have ACLs for artifacts sharing such as notebook content, model, etc. |
| Admin | Also called SRE, who manages user's quotas, credentials, team, and other components. |
| Project | A project may include one or multiple notebooks, zero or multiple running jobs. And could be collaborated by multiple users who have ACLs on it |


# Background 

Everybody talks about machine learning today, and lots of companies are trying to leverage machine learning to push the business to the next level. Nowadays, as more and more developers, infrastructure software companies coming to this field, machine learning becomes more and more archivable. 

In the last decade, software industry has built many open source tools for machine learning to solve the pain points: 

1. It was not easy to build machine learning algorithms manually such as logistic regression, GBDT, and many other algorithms:
   **Answer to that:** Industries have open sourced many algorithm libraries, tools and even pre-trained models so that data scientists can directly reuse these building blocks to hook up to their data without knowing intricate details inside these algorithms and models. 

2. It was not easy to achieve "WYSIWYG, what you see is what you get" from IDEs: not easy to get output, visualization, troubleshooting experiences at the same place. 
   **Answer to that:** Notebooks concept was added to this picture, notebook brought the experiences of interactive coding, sharing, visualization, debugging under the same user interface. There're popular open-source notebooks like Apache Zeppelin/Jupyter.
   
3. It was not easy to manage dependencies, ML applications can run on one machine is hard to deploy on another machine because it has lots of libraries dependencies. 
   **Answer to that:** Containerization becomes popular and a standard to packaging dependencies to make it easier to "build once, run anywhere". 

4. Fragmented tools, libraries were hard for ML engineers to learn. Experiences learned in one company is not naturally migratable to another company.
   **Answer to that:** A few dominant open-source frameworks reduced the overhead of learning too many different frameworks, concept. Data-scientist can learn a few libraries such as Tensorflow/PyTorch, and a few high-level wrappers like Keras will be able to create your machine learning application from other open-source building blocks.

4. Similarly, models built by one library (such as libsvm) were hard to be integrated to machine learning pipeline since there's no standard format.
   **Answer to that:** Industry has built successful open-source standard machine learning frameworks such as Tensorflow/PyTorch/Keras so their format can be easily shared across. And efforts to build a even more general model format such as ONNX.
   
5. It was hard to build a data pipeline which flows/transform data from raw data source to whatever required by ML applications. 
   **Answer to that:** Open source big data industry plays an important role to provide, simplify, unify processes and building blocks for data flows, transformations, etc.
   
Machine learning industry is moving on the right track to solve major roadblocks. So what is the pain points now for companies who have machine learning needs? What we can help here? To answer this question, let's look at machine learning workflow first. 

## Machine Learning Workflows & Pain points

```
1) From different data source such as edge, clickstream, logs, etc.
   => Land to data lakes  
   
2) From data lake, data transformation: 
   => Data transformations: Cleanup, remove invalid rows/columns, 
                            select columns, sampling, split train/test
                            data-set, join table, etc.
   => Data prepared for training.
                            
3) From prepared data: 
   => Training, model hyper-parameter tuning, cross-validation, etc. 
   => Models saved to storage. 
   
4) From saved models: 
   => Model assurance, deployment, A/B testing, etc.
   => Model deployed for online serving or offline scoring.
```

Typically data scientists responsible for item 2)-4), 1) typically handled by a different team (called Data Engineering team in many companies, some Data Engineering team also responsible for part of data transformation)

### Pain \#1 Complex workflow/steps from raw data to model, many different tools need by different steps, hard to make changes to workflow, and not error-proof

It is a complex workflow from raw data to usable models, after talking to many different data scientists, we have learned that a typical procedure to train a new model and push to production can take months to 1-2 years. 

It is also a wide skill set required by this workflow. For example, data transformation needs tools like Spark/Hive for large scale and tools like Pandas for small scale. And model training needs to be switched between XGBoost, Tensorflow, Keras, PyTorch. Building a data pipeline needs Apache Airflow or Oozie. 

Yes, there are great, standardized open-source tools built for many of such purposes. But how about changes need to be made for a particular part of the data pipeline? How about adding a few columns to the training data for experiments? How about training models, and push models to validation, A/B testing before rolling to production? All these steps need jumping between different tools, UIs, and very hard to make changes, and it is not error-proof during these procedures.

### Pain \#2 Dependencies of underlying resource management platform

To make jobs/services required by machine learning platform to be able to run, we need an underlying resource management platform. There're some choices of resource management platform and they have distinct advantages and disadvantages. 

For example, there're many machine learning platform built on top of K8s. It is relatively easy to get a K8s from a cloud vendor, easy to orchestrate machine learning required services/daemons run on K8s. However, K8s doesn't offer good support jobs like Spark/Flink/Hive. So if your company has Spark/Flink/Hive running on YARN, there're gaps and a significant amount of work to move required jobs from YARN to K8s. Maintaining a separate K8s cluster is also overhead to Hadoop-based data infrastructure.

Similarly, if your company's data pipelines are mostly built on top of cloud resources and SaaS offerings. Asking you to install a separate YARN cluster to run a new machine learning platform doesn't make a lot of sense.

### Pain \#3 Data scientist are forced to interact with lower-level platform components

In addition to the above pain, we do see Data Scientists are forced to learn underlying platform knowledge to be able to build a real-world machine learning workflow.

For most of the data scientists we talked with, they're experts of ML algorithms/libraries, feature engineering, etc. They're also most familiar with Python, R, and some of them understand Spark, Hive, etc. 

If they're asked to do interactions with lower-level components like fine-tuning a Spark job's performance; or troubleshooting job failed to launch because of resource constraints; or write a K8s/YARN job spec and mount volumes, set networks properly. They will scratch their heads and typically cannot perform these operations efficiently.

### Pain \#4 Comply with data security/governance requirements

TODO: Add more details.

### Pain \#5 No good way to reduce routine ML code development

After the data is prepared, the data scientist needs to do several routine tasks to build the ML pipeline. To get a sense on the existing data set, it usually needs a split of the data set, the statistics of data set. These tasks have a common duplicate part of code which reduces the efficiency of data scientists.

An abstraction layer/framework to help developer to boost ML pipeline development could be valuable. It's better that the developer only needs to fill callback function to focus on their key logics.

# Submarine

## Overview

### A little bit history

Initially, Submarine is built to solve problems of running deep learning jobs like Tensorflow/PyTorch on Apache Hadoop YARN, allows admin to monitor launched deep learning jobs, and manage generated models. 

It was part of YARN initially, code resides under `hadoop-yarn-applications`. Later, the community decided to move to a subject of Hadoop because we want to support other resource management platforms like K8s. And finally, we're reconsidering Submarine's charter and Hadoop community voted that it is the time to moved Submarine to a separate Apache TLP.

### Why Submarine? 

`ONE PLATFORM`

Submarine is the ONE PLATFORM to allow Data Scientists to create end-to-end machine learning workflow. `ONE PLATFORM` means it supports Data Scientists and data engineers to finish their jobs on the same platform without frequently switching their toolsets. From dataset exploring to data pipeline creation, model training, and tuning, and push model to production. All these steps can be completed within the `ONE PLATFORM`.

`Resource Management Independent`

It is also designed to be resource management independent, no matter if you have Apache Hadoop YARN, K8s, or just a container service, you will be able to run Submarine on top it.


## Requirements and non-requirements

### Requirements

Following items are charters of Submarine project:

#### Notebook

1) Users should be able to create, edit, delete a notebook. (P0)
2) Notebooks can be persisted to storage and can be recovered if failure happens. (P0)
3) Users can trace back to history versions of a notebook. (P1)
4) Notebook can be shared with different users. (P1)
5) Users can define a list of parameters of a notebook (looks like parameters of notebook's main function) to allow execute a notebook like a job. (P1)
6) Different users can collaborate on the same notebook at the same time. (P2)

#### Job

Job of Submarine is an executable code section. It could be a shell command, a Python command, a Spark job, a SQL query, a training job (such as Tensorflow), etc. 

1) Job can be submitted from UI/CLI.
2) Job can be monitored/managed from UI/CLI.
3) Job should not bind to one resource management platform (YARN/K8s).

#### Training Job

Training job is a special kind of job, which includes Tensorflow, PyTorch and other different frameworks: 

1) Allow model engineer, data scientist to run *unmodified* Tensorflow programs on YARN/K8s/Container-cloud. 
2) Allow jobs easy access data/models in HDFS and other storages. 
3) Support run distributed Tensorflow jobs with simple configs.
4) Support run user-specified Docker images.
5) Support specify GPU and other resources.
6) Support launch tensorboard (and other equivalents for non-TF frameworks) for training jobs if user specified.

[TODO] (Need help)

#### Model Management 

After training, there will be model artifacts created. Users should be able to:

1) View model metrics.
2) Save, versioning, tagging model.
3) Run model verification tasks. 
4) Run A/B testing, push to production, etc.

#### Metrics for training job and model

Submarine-SDK provides tracking/metrics APIs which allows developers add tracking/metrics and view tracking/metrics from Submarine Workbench UI.

#### Workflow 

Data-Scientists/Data-Engineers can create workflows from UI. Workflow is DAG of jobs.

### Non-requirements

TODO: Add non-requirements which we want to avoid.

[TODO] (Need help)

## Architecture Overview

### Architecture Diagram

```
      +-----------------<---+Submarine Workbench+---->------------------+
      | +---------+ +---------+ +-----------+ +----------+ +----------+ |
      | |Data Mart| |Notebooks| |Projects   | |Metrics   | |Models    | |
      | +---------+ +---------+ +-----------+ +----------+ +----------+ |
      +-----------------------------------------------------------------+


      +----------------------Submarine Service--------------------------+
      |                                                                 |
      | +-----------------+ +-----------------+ +--------------------+  |
      | |Compute Engine   | |Job Orchestrator | |     SDK            |  |
      | |    Connector    | |                 | +--------------------+  |
      | +-----------------+ +-----------------+                         |
      |   Spark, Flink         YARN/K8s/Docker    Java/Python/REST      |
      |   TF, PyTorch                               Mini-Submarine      |
      |                                                                 |
      |                                                                 |
      +-----------------------------------------------------------------+
      
      (You can use http://stable.ascii-flow.appspot.com/#Draw 
      to draw such diagrams)
```

#### Submarine Workbench 

Submarine Workbench is a UI designed for data scientists. Data scientists can interact with Submarine Workbench UI to access notebooks, submit/manage jobs, manage models, create model training workflows, access dataset, etc.

### Components for Data Scientists

1) `Notebook Service` helps to do works from data insight to model creation and allows notebook sharing between teams. 
2) `Workflow Service` helps to construct workflows across notebook or include other executable code entry point. This module can construct a DAG and execute user-specified workflow from end to end. 

`NoteBook Service` and `Workflow Service` deployed inside Submarine Workbench Server, and provides Web, CLI, REST APIs for 3rd-party integration.

4) `Data Mart` helps to create, save and share dataset which can be used by other modules or training.
5) `Model Training Service`
   - `Metrics Service` Helps to save metrics during training and analysis training result if needed.
   - `Job Orchestrator` Helps to submit a job (such as Tensorflow/PyTorch/Spark) to a resource manager, such as YARN or K8s. It also supports submit a distributed training job. Also, get status/logs, etc. of a job regardless of Resource Manager implementation. 
   - `Compute Engine Connector` Work with Job Orchestrator to submit different kinds of jobs. One connector connects to one specific kind of compute framework such as Tensorflow. 
6) `Model Service` helps to manage, save, version, analysis a model. Also helps to push model to production.
7) `Submarine SDK` provides Java/Python/REST API to allow DS or other engineers to integrate to Submarine services. It also includes a `mini-submarine` component which launches Submarine components from a single Docker container (or a VM image).
8) `Project Manager` helps to manage projects. Each project can have multiple notebooks, workflows, etc.

### Components for SREs 

Following components are designed for SREs (or system admins) of the Machine Learning Platform.

1) `User Management System` helps admin to onboard new users, upload user credentials, assign resource quotas, etc. 
2) `Resource Quota Management System` helps admin to manage resources quotas of teams, organizations. Resources can be machine resources like CPU/Memory/Disk, etc. It can also include non-machine resources like $$-based budgets.

[TODO] (Need help)

## User Flows

### User flows for Data-Scientists/Data-engineers

DS/DE will interact with Submarine to do the following things: 

New onboard to Submarine Service:
- Need Admin/SRE help to create a user account, set up user credentials (to access storage, metadata, resource manager, etc.), etc.
- Submarine can integrate with LDAP or similar systems, users can login use OpenID, etc.

Access Data: 
- DS/DE can access data via DataMart. DataMart is an abstraction layer to view available datasets which can be accessed. DS/DE need proper underlying permission to read this data. 
- DataMart provides UIs/APIs to allow DS/DE to preview, upload, import data-sets from various locations.
- DS/DE can also bring their data from different sources, on-prem or on-cloud. They can also add a data quick link (like an URL) to DataMart.
- Data access could be limited by the physical location of compute clusters. For example, it is not viable to access data stored in another data center which doesn't have network connectivity setup.

Projects: 
- Every user starts with a default project. User can choose to create new projects.
- A project belongs to one user, user can share project with different users/teams.
- User can clone a project belongs to other users who have access.
- A project can have notebooks, dependencies, or any required custom files. (Such as configuration files). We don't suggest to upload any full-sized data-set files to a project.
- Projects can include folders (It's like a regular FS), and folders can be mounted to a running notebook/jobs.

Notebook: 
- A notebook belongs to a project.
- User can create, clone, import (from file), export (to file) notebook. 
- User can ask to attach notebook by notebook service on one of the compute cluster. 
- In contrast, user can ask to detach a running notebook instance.
- A notebook can be shared with other users, teams.
- A notebook will be versioned, persisted (using Git, Mysql, etc.), and user can traverse back to older versions.

Dependencies:
- A dependency belongs to a project.
- User can add dependencies (a.k.a libraries) to a project. 
- Dependency can be jar/python(PyPI, Egg, Whil), etc.
- User can choose to have BYOI (bring your own container image) for a project. 

Job: 
- A job belongs to a project. 
- User can run (or terminate) job with type and parameters on one of the running cluster.
- User can get the status of running jobs, retrieve job logs, metrics, etc.
- Job submission and basic operation should be available on both API (CLI) and UI. 
- For different types of jobs, Submarine's `Compute Engine Connector` allows taking different parameters to submit a job. For example, submitting a `Tensorflow` job allows a specifying number of parameter servers, workers, etc. Which is different from `Spark` job.
- Notebook can be treated as a special kind of job. (Runnable notebook).

Workflow: 
- A workflow belongs to a project. 
- A workflow is a DAG of jobs. 
- User can submit/terminate a workflow.
- User can get status from a workflow, and also get a list of running/finished/failed/pending jobs from the workflow.
- Workflow can be created/submitted via UI/API.

Model:
- Model generated by training jobs.
- A model consists of artifacts from one or multiple files. 
- User can choose to save, tag, version a produced model.
- Once model is saved, user can do online serving or offline scoring of the model.

### User flows for Admins/SRE

Operations for users/teams: 
- Admins can create new user, new team, update user/team mappings. Or remove users/teams. 
- Admin can set resource quotas (if different from system default), permissions, upload/update necessary credentials (like kerberos keytab) of a user.
- A DE/DS can also be an admin if the DE/DS have admin access. (Like a privileged user). This will be useful when a cluster is exclusively shared by a user or only shared by a small team.

## Deployment

```


    +---------------Submarine Service---+
    |                                   |
    | +------------+ +------------+     |
    | |Web Svc/Prxy| |Backend Svc |     |    +--Submarine Data+-+
    | +------------+ +------------+     |    |Project/Notebook  |
    |   ^                               |    |Model/Metrics     |
    +---|-------------------------------+    |Libraries/DataMart|
        |                                    +------------------+
        |
        |      +----Compute Cluster 1---+    +--Image Registry--+
        +      |User Notebook Instance  |    |   User's Images  |
      User /   |Jobs (Spark, TF)        |    |                  |
      Admin    |                        |    +------------------+
               |                        |
               +------------------------+    +-Data Storage-----+
                                             | S3/HDFS, etc.    |
               +----Compute Cluster 2---+    |                  |
                                             +------------------+
                        ...
```

Here's a diagram to illustrate Submarine's deployment.

- Submarine service consists of web service/proxy, and backend services. They're like "control planes" of Submarine, and users will interact with these services.
- Submarine service could be a microservice architecture and can be deployed to one of the compute clusters. (see below). 
- There're multiple compute clusters could be used by Submarine service. For user's running notebook instance, jobs, etc. they will be placed to one of the compute cluster by user's preference or defined policies.
- Submarine's data includes project/notebook(content)/models/metrics, etc. will be stored separately from dataset (DataMart)
- Datasets can be stored in various locations such as S3/HDFS. 
- User can push container (such as Docker) images to a preconfigured registry in Submarine so Submarine service can know how to pull required container images.


## Security Models

[TODO] (Need help)

# References
