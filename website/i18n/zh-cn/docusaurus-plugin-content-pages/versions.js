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

import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {
  useVersions,
  useLatestVersion,
} from '@docusaurus/plugin-content-docs/client';

import versionsReleaseNoteOnly from '@site/versionsReleaseNoteOnly.json';

function Version() {
  const versions = useVersions();
  const latestVersion = useLatestVersion();
  const {
    siteConfig
  } = useDocusaurusContext();
  const currentVersion = versions.find((version) => version.name === 'current');
  const pastVersions = versions.filter(
    (version) => version !== latestVersion && version.name !== 'current',
  );
  const repoUrl = `https://github.com/${siteConfig.organizationName}/${siteConfig.projectName}`;
  return (
    <Layout
      title="Versions"
      description="Submarine Versions page listing all documented site versions">
      <main className="container margin-vert--lg">
        <h1>Submarine 文档版本</h1>

        {latestVersion && (
          <div className="margin-bottom--lg">
            <h3 id="next">当前版本 (稳定版)</h3>
            <p>
              在这里，您可以找到当前发布版本的文档。
            </p>
            <table>
              <tbody>
                <tr>
                  <th>{latestVersion.label}</th>
                  <td>
                    <Link to={latestVersion.path + "/" + latestVersion.mainDocId}>文档</Link>
                  </td>
                  <td>
                    <a href={`./releases/submarine-release-${latestVersion.name}`}>
                      发布公告
                    </a>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {currentVersion !== latestVersion && (
          <div className="margin-bottom--lg">
            <h3 id="latest">下一版本 (未发布)</h3>
            <p>
              在这里，您可以找到未发布版本的文档。
            </p>
            <table>
              <tbody>
                <tr>
                  <th>{currentVersion.label}</th>
                  <td>
                    <Link to={currentVersion.path + "/" + currentVersion.mainDocId}>文档</Link>
                  </td>
                  <td>
                    <Link to={repoUrl}>源代码</Link>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {(pastVersions.length + versionsReleaseNoteOnly.length) > 0 && (
          <div className="margin-bottom--lg">
            <h3 id="archive">过去的版本 (不再维护)</h3>
            <p>
              在这里你可以找到以前版本的 Submarine 文档。
            </p>
            <table>
              <tbody>
                {pastVersions.map((version) => (
                  <tr key={version.name}>
                    <th>{version.label}</th>
                    <td>
                      <Link to={version.path + "/" + version.mainDocId}>文档</Link>
                    </td>
                    <td>
                      <a href={`./releases/submarine-release-${version.name}`}>
                        发布公告
                      </a>
                    </td>
                  </tr>
                ))}
                {versionsReleaseNoteOnly.map((version) => (
                  <tr key={version}>
                    <th>{version}</th>
                    <td>
                      文档
                    </td>
                    <td>
                      <a href={`./releases/submarine-release-${version}`}>
                        发布公告
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </Layout>
  );
}

export default Version;
