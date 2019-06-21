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
package org.apache.submarine.tony.models;

import org.apache.submarine.tony.Constants;
import org.apache.submarine.tony.util.Utils;
import java.util.Date;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.conf.YarnConfiguration;


public class JobMetadata {
  private String id;
  private String jobLink;
  private String configLink;
  private String rmLink;
  private long started;
  private long completed;
  private String status;
  private String user;

  private JobMetadata(JobMetadata.Builder builder) {
    this.id = builder.id;
    this.jobLink = "/" + Constants.JOBS_SUFFIX + "/" + id;
    this.configLink = "/" + Constants.CONFIG_SUFFIX + "/" + id;
    this.rmLink = Utils.buildRMUrl(builder.conf, builder.id);
    this.started = builder.started;
    this.completed = builder.completed;
    this.status = builder.status;
    this.user = builder.user;
  }

  public static JobMetadata newInstance(YarnConfiguration conf, String histFileName) {
    String histFileNoExt = histFileName.substring(0, histFileName.indexOf('.'));
    String[] metadata = histFileNoExt.split("-");
    Builder metadataBuilder = new Builder()
        .setId(metadata[0])
        .setStarted(Long.parseLong(metadata[1]))
        .setConf(conf);
    if (histFileName.endsWith(Constants.INPROGRESS)) {
      metadataBuilder
          .setUser(metadata[2])
          .setStatus(Constants.RUNNING);
      return metadataBuilder.build();
    }
    metadataBuilder
        .setCompleted(Long.parseLong(metadata[2]))
        .setUser(metadata[3])
        .setStatus(metadata[4]);
    return metadataBuilder.build();
  }

  public static class Builder {
    private String id = "";
    private long started = -1L;
    private long completed = -1L;
    private String status = "";
    private String user = "";
    private Configuration conf = null;

    public JobMetadata build() {
      return new JobMetadata(this);
    }

    public JobMetadata.Builder setId(String id) {
      this.id = id;
      return this;
    }

    public JobMetadata.Builder setConf(Configuration conf) {
      this.conf = conf;
      return this;
    }

    public JobMetadata.Builder setStarted(long startTime) {
      this.started = startTime;
      return this;
    }

    public JobMetadata.Builder setCompleted(long completed) {
      this.completed = completed;
      return this;
    }

    public JobMetadata.Builder setStatus(String status) {
      this.status = status;
      return this;
    }

    public JobMetadata.Builder setUser(String user) {
      this.user = user;
      return this;
    }
  }

  public String getId() {
    return id;
  }

  public String getJobLink() {
    return jobLink;
  }

  public String getConfigLink() {
    return configLink;
  }

  public String getRMLink() {
    return rmLink;
  }

  public Date getStartedDate() {
    return new Date(started);
  }

  public long getStarted() {
    return started;
  }

  public Date getCompletedDate() {
    return new Date(completed);
  }

  public long getCompleted() {
    return completed;
  }

  public String getStatus() {
    return status;
  }

  public String getUser() {
    return user;
  }

  public void setId(String id) {
    this.id = id;
  }

  public void setUser(String user) {
    this.user = user;
  }
}
