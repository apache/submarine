<!---
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

`crd.yaml`

This file defines the schema of our [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/).

`example-submarine.yaml`

This file specifies the metadata of our custom resource, we can apply this file to deploy it.

`submarine-operator-service-account.yaml`

This file grants that submarine-operator has cluster-wide access, which allow it to operate on other resources.

`submarine-operator.yaml`

We can apply this file to deploy submarine-operator.





