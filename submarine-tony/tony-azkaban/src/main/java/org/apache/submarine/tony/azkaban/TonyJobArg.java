/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

package org.apache.submarine.tony.azkaban;

// Standard TonY job arguments.
public enum TonyJobArg {
  HDFS_CLASSPATH("hdfs_classpath"),
  SHELL_ENV("shell_env"),
  TASK_PARAMS("task_params"),
  PYTHON_BINARY_PATH("python_binary_path"),
  PYTHON_VENV("python_venv"),
  EXECUTES("executes"),
  SRC_DIR("src_dir");

  TonyJobArg(String azPropName) {
    this.azPropName = azPropName;
    this.tonyParamName = "-" + azPropName;
  }

  final String azPropName;
  final String tonyParamName;
}
