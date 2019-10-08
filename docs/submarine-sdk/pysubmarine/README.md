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

# PySubmarine
PySubmarine helps developers use submarine's internal data caching,
data exchange, and task tracking capabilities to more efficiently improve the 
development and execution of machine learning productivity.

## Package setup
- Clone repo
```bash
git clone https://github.com/apache/hadoop-submarine.git 
cd hadoop-submarine/submarine-sdk/pysubmarine
```

- Install pip package
```bash
pip install .
```

- Run tests
```bash
pytest --cov=submarine -vs
```

- Run checkstyle
```bash
pylint --msg-template="{path} ({line},{column}): \
[{msg_id} {symbol}] {msg}" --rcfile=pylintrc -- submarine tests
```

## PySubmarine API Reference
### Tracking
- [Tracking](tracking.md)