# Submarine Cluster Server Design

## Introduction
The Submarine system contains a total of two daemon services, Submarine Server and Workbench Server.

Submarine Server mainly provides job submission, job scheduling, job status monitoring, and model online service for Submarine.

Workbench Server is mainly for algorithm users to provide algorithm development, Python/Spark interpreter operation and other services through Notebook.

The goal of the Submarine project is to provide high availability and high reliability services for big data processing, 
algorithm development, job scheduling, model online services, model batch and incremental updates. 

In addition to the high availability of big data and machine learning frameworks, 
the high availability of Submarine Server and Workbench Server itself is a key consideration.

## Requirement

### Cluster Metadata Center

Multiple Submarine (or Workbench) Server processes create a Submarine Cluster through the RAFT algorithm library. 

The cluster internally maintains a metadata center. All servers can operate the metadata. 

The RAFT algorithm ensures that multiple processes are simultaneously co-located. 

A data modification will not cause problems such as mutual coverage and dirty data.

This metadata center stores data by means of key-value pairs. it can store/support a variety of data, 
but it should be noted that metadata is only suitable for storing small amounts of data and cannot be used to replace data storage.

### Service discovery

By storing the information of the service or process in the metadata center, we can easily find the information of the service or process we need in any place, 
for example, the IP address and port where the Python interpreter will be the process. Information is stored in metadata, 
and other services can easily find process information through process IDs and connect to provide service discovery capabilities.

### Cluster event

In the entire Submarine cluster, the servers can communicate with each other and other child processes to send cluster events to each other. 

The service or process processes the corresponding programs according to the cluster events. For example, 
the Workbench Server can be managed to Python. The interpreter process sends a shutdown event that controls the operation of the services and individual subprocesses throughout the cluster.

Cluster events support both broadcast and separate delivery capabilities.

### Independence

We implement Submarine's clustering capabilities through the RAFT algorithm library, without relying on any external services (eg Zookeeper, Etcd, etc.)

### Disadvantages

Because the RAFT algorithm requires more than half of the servers available to ensure the normality of the RAFT algorithm, 
if we need to turn on the clustering capabilities of Submarine (Workbench) Server, when more than half of the servers are unavailable, 
some programs may appear abnormal. Of course, we also detected this in the system, downgrading the system or refusing to provide service status.

## System design

### Universal design

Modular design, because Submarine (Workbench) Server exists in Submarine system, these two services need to provide clustering capabilities, 
so we abstract the cluster function into a separate module for development, so that Submarine (Workbench) Server can reuse the cluster function module.

### ClusterConfigure

Add a `submarine.server.addr` and `workbench.server.addr` configuration items in `submarine-site.xml`, `submarine.server.addr=ip1, ip2, ip3`, 
through the IP list, the RAFT algorithm module in the server process can Cluster with other server processes.

### ClusterServer

+ The ClusterServer module encapsulates the RAFT algorithm module, which can create a service cluster and read and write metadata based 
on the two configuration items submarine.server.addr or workbench.server.addr.

+ The cluster management service runs in each submarine server;

+ The cluster management service establishes a cluster by using the atomix RaftServer class of the Raft algorithm library, maintains the ClusterStateMachine, 
and manages the service state metadata of each submarine server through the PutCommand, GetQuery, and DeleteCommand operation commands.

### ClusterClient

+ The ClusterClient module encapsulates the RAFT algorithm client module, which can communicate with the cluster according to the two configuration items `submarine.server.addr` or `workbench.server.addr`, 
read and write metadata, and write the IP and port information of the client process. Into the cluster's metadata center.

+ The cluster management client runs in each submarine server and submarine Interpreter process;

+ The cluster management client manages the submarine server and submarine Interpreter process state (metadata information) 
in the ClusterStateMachine by using the atomix RaftClient class of the Raft library to connect to the atomix RaftServer. 

+ When the submarine server and Submarine Interpreter processes are started, they are added to the ClusterStateMachine and are removed from the ClusterStateMachine 

+ when the Submarine Server and Submarine Interpreter processes are closed.

### ClusterMetadata
Metadata stores metadata information in a KV key-value pair。
ServerMeta：key='host:port'，value= {SERVER_HOST=...，SERVER_PORT=...，...}



| Name                  | Description                     |
| --------------------- | ------------------------------- |
| SUBAMRINE_SERVER_HOST | Submarine server IP             |
| SUBAMRINE_SERVER_PORT | Submarine server port           |
| WORKBENCH_SERVER_HOST | Submarine workbench server IP   |
| WORKBENCH_SERVER_PORT | Submarine workbench server port |

InterpreterMeta：key=InterpreterGroupId，value={INTP_TSERVER_HOST=...，...}

| Name              | Description                          |
| ----------------- | ------------------------------------ |
| INTP_TSERVER_HOST | Submarine Interpreter Thrift IP      |
| INTP_TSERVER_PORT | Submarine Interpreter Thrift port    |
| INTP_START_TIME   | Submarine Interpreter start time     |
| HEARTBEAT         | Submarine Interpreter heartbeat time |

### Network fault tolerance

In a distributed environment, there may be network anomalies, network delays, or service exceptions. After submitting metadata to the cluster, 
check whether the submission is successful. After the submission fails, save the metadata in the local message queue. a separate commit thread to retry;

### Cluster monitoring

The cluster needs to monitor whether the Submarine Server and Submarine-Interpreter processes are working properly.

The Submarine Server and Submarine Interpreter processes periodically send heartbeats to update their own timestamps in the cluster metadata. 

The Submarine Server with Leader identity periodically checks the timestamps of the Submarine Server and Submarine Interpreter processes to clear the timeout services and processes.

1. The cluster monitoring module runs in each Submarine Server and Submarine Interpreter process, 
periodically sending heartbeat data of the service or process to the cluster;

2. When the cluster monitoring module runs in Submarine Server, it sends the heartbeat to the cluster's ClusterStateMachine. 
If the cluster does not receive heartbeat information for a long time, Indicates that the service or process is abnormal and unavailable.

3. Resource usage statistics strategy, in order to avoid the instantaneous high peak and low peak of the server, 
the cluster monitoring will collect the average resource usage in the most recent period for reporting, and improve the reasonable line and effectiveness of the server resources as much as possible;

4. When the cluster monitoring module runs in Submarine Server, it checks the heartbeat data of each Submarine Server and Submarine Interpreter process. 
If it times out, it considers that the service or process is abnormally unavailable and removes it from the cluster.

### Atomix Raft algorithm library

In order to reduce the deployment complexity of distributed mode, submarine server does not use zookeeper to build a distributed cluster. 
Multiple submarine server groups are built into distributed clusters by using the Raft algorithm in submarine server. 
The Raft algorithm is involved by atomix lib of atomix that has passed Jepsen consistency verification.

### Synchronize workbench notes

In cluster mode, the user creates, modifies, and deletes the note on any of the servers. 
all need to be notified to all the servers in the cluster to synchronize the update of Notebook. 
failure to do so will result in the user not being able to continue while switching to another server.

### Listen for note update events

Listen for the NEW_NOTE, DEL_NOTE, REMOVE_NOTE_TO_TRASH ... event of the notebook in the NotebookServer#onMessage() function.

### Broadcast note update event

The note is refreshed by notifying the event to all Submarine servers in the cluster via messagingService.
