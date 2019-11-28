/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.submarine.client.cli.param.runjob;

import com.google.common.collect.Lists;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.submarine.client.cli.CliConstants;
import org.apache.submarine.client.cli.param.ParametersHolder;
import org.apache.submarine.client.cli.param.Quicklink;

import java.util.List;

/**
 * Parses quicklink parameters (if any) into a {@link QuickLinks} object.
 */
public final class QuickLinks {
  private List<Quicklink> links;

  public QuickLinks(List<Quicklink> links) {
    this.links = links;
  }

  public static QuickLinks parse(ParametersHolder parametersHolder)
      throws YarnException, ParseException {
    List<String> quicklinkStrs = parametersHolder.getOptionValues(
        CliConstants.QUICKLINK);

    List<Quicklink> links = Lists.newArrayList();
    if (quicklinkStrs != null) {
      for (String linkStr : quicklinkStrs) {
        Quicklink quicklink = new Quicklink();
        quicklink.parse(linkStr);
        links.add(quicklink);

      }

    }
    return new QuickLinks(links);
  }

  public List<Quicklink> getLinks() {
    return links;
  }
}