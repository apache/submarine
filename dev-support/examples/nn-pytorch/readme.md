# Save_model Example

## Usage
This is an easy example of saving a pytorch linear model to model registry.

## How to execute

1. Build the docker image

```bash
./dev-support/examples/nn-pytorch/build.sh
```

2. Submit a post request

```bash
./dev-support/examples/nn-pytorch/post.sh
```

## Serve the model by Serve API

1. Make sure the model is saved in the model registry (viewed on MLflow UI)
2. Call serve API to create serve resource
- Request
  ```
  curl -X POST -H "Content-Type: application/json" -d '
  {
    "modelName":"simple-nn-model",
    "modelVersion":"1",
    "namespace":"default"
  }
  ' http://127.0.0.1:32080/api/v1/experiment/serve
  ```
- Response
  ```
  {
      "status": "OK",
      "code": 200,
      "success": true,
      "message": null,
      "result": {
          "url": "/serve/simple-nn-model-1"
      },
      "attributes": {}
  }
  ```

3. Send data to be inferenced
- Request
  ```
  curl -d '{"data":[[-1, -1]]}' -H 'Content-Type: application/json; format=pandas-split' -X POST http://127.0.0.1:32080/serve/simple-nn-model-1/invocations
  ```
- Response
  ```
  [{"0": -0.5663654804229736}]
  ```
4. Call serve API to delete serve resource
- Request
  ```
  curl -X DELETE http://0.0.0.0:32080/api/v1/experiment/serve?modelName=simple-nn-model&modelVersion=1&namespace=default
  ```
- Response
  ```
  {"status":"OK","code":200,"success":true,"message":null,"result":{"url":"/serve/simple-nn-model-1"},"attributes":{}}
  ```