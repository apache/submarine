---
title: Security Implementation
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

:::warning
Please note that this design doc is working-in-progress and need more works to complete. 
:::

## Handle User's Credential

Users credential includes Kerberoes Keytabs, Docker registry credentials, Github ssh-keys, etc.

User's credential must be stored securitely, for example, via KeyCloak or K8s Secrets.

(More details TODO)