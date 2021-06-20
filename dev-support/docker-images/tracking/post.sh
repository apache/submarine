#!/usr/bin/env bash
curl -X POST -H "Content-Type: application/json" -d '
{
  "meta": {
    "name": "tracking-example2",
    "namespace": "default",
    "framework": "TensorFlow",
    "cmd": "python /opt/tracking.py",
    "envVars": {
      "ENV_1": "ENV1"
    }
  },
  "environment": {
    "image": "tracking:0.6.0-SNAPSHOT"
  },
  "spec": {
    "Ps": {
      "replicas": 1,
      "resources": "cpu=1,memory=128M"
    },
    "Worker": {
      "replicas": 3,
      "resources": "cpu=1,memory=128M"
    }
  }
}
' http://127.0.0.1:32080/api/v1/experiment