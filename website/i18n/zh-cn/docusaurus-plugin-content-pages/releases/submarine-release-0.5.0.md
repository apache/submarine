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

# Apache Submarine Release 0.5.0

Apache Submarine 社区很高兴地宣布 `0.5.0` 版本已经发布。

自上次发布以来，社区投入了大量精力来改进 Apache Submarine。
总计有 99 个用于改进和错误修复的补丁。  
新特性如下：

- Submarine Experiments: 重新定义 experiment 的 spec，可以使用 HTTP 和 ssh 从 Git 同步代码
- 预定义的 experiment 模板: 注册一个 experiment 模板并提交相关参数以使用 Rest API 来运行 experiment
- Environment 配置: 用户可以轻松管理他们的 docker 镜像和 conda 环境
- Jupyter Notebook: 使用 Rest API 生成 jupyter notebook，并在 K8s 上执行 ML 代码，或提交 experiment 到 submarine server
- Submarine Workbench UI: 通过 UI 进行 Experiment、Environment、Notebook 的 CRUD 操作
- 禁用 interpreter 模块


我们推荐 [下载](../docs/download) 最新版本进行体验和使用，并非常欢迎通过 [邮件列表](../docs/community/) 提供反馈。

您还可以访问 [问题跟踪器](https://issues.apache.org/jira/secure/ReleaseNote.jspa?version=12348041&projectId=12322824) 以获取已解决问题的完整列表。
