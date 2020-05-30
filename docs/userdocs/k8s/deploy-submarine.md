

# Deploy Submarine On K8s

## Deploy Submarine using Helm Chart (Recommended)

### Create images
submarine server
```bash
./dev-support/docker-images/submarine/build.sh
```

submarine database
```bash
./dev-support/docker-images/database/build.sh
```

### install helm
For more info see https://helm.sh/docs/intro/install/

### Deploy Submarine Server, mysql
You can modify some settings in ./helm-charts/submarine/values.yaml
```bash
helm install submarine ./helm-charts/submarine
```

### Delete deployment
```bash
helm delete submarine
```

FIXME: TF Operator / PyTorch Operator can be deployed by Helm Chart, correct?

## Deploy Submarine Manually

### Get package
You can dowload submarine from releases page or build from source.

### Configuration
Copy the kube config into `conf/k8s/config` or modify the `conf/submarine-site.xml`:
```
<property>
  <name>submarine.k8s.kube.config</name>
  <value>PATH_TO_KUBE_CONFIG</value>
</property>
```

### Start Submarine Server
Running the submarine server, executing the following command:
```
# if build from source. You need to run this under the target dir like submarine-dist/target/submarine-dist-0.4.0-SNAPSHOT-hadoop-2.9/submarine-dist-0.4.0-SNAPSHOT-hadoop-2.9/
./bin/submarine-daemon.sh start getMysqlJar
```

The REST API URL is: `http://127.0.0.1:8080/api/v1/jobs`

### Deploy Tensorflow Operator
For more info see [deploy tensorflow operator](./ml-frameworks/tensorflow.md).

### Deploy PyTorch Operator
```bash
cd <submarine_code_path_root>/dev-support/k8s/pytorchjob
./deploy-pytorch-operator.sh

```

## Post Deploy

### Setup port-forward {host port}:{container port}

FIXME: Why we need this? Need elaborate

```bash
kubectl port-forward svc/submarine-server 8080:8080 --address 0.0.0.0
```

