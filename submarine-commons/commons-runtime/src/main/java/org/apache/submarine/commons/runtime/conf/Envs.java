/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.submarine.commons.runtime.conf;

public class Envs {
  public static final String TASK_TYPE_ENV = "_TASK_TYPE";
  public static final String TASK_INDEX_ENV = "_TASK_INDEX";

  /*
   * HDFS/HADOOP-related configs
   */
  public static final String HADOOP_HDFS_HOME = "HADOOP_HDFS_HOME";
  public static final String JAVA_HOME = "JAVA_HOME";
  public static final String HADOOP_CONF_DIR = "HADOOP_CONF_DIR";
}
