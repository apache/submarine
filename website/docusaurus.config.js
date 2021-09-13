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

module.exports = {
  title: 'Apache Submarine',
  tagline: 'Cloud Native Machine Learning Platform',
  url: 'https://submarine.apache.org/',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/submarine.ico',
  organizationName: 'apache', // Usually your GitHub org/user name.
  projectName: 'submarine-site', // Don't change the project name, the website will be updated to submarine-site repo
  themeConfig: {
    navbar: {
      title: 'Apache Submarine',
      logo: {
        alt: 'Apache Submarine Site Logo',
        src: 'img/icons/128.png',
      },
      items: [
        {
          type: 'doc',
          docId: 'gettingStarted/quickstart',
          label: 'Docs',
          position: 'left',
        },
        {
          type: 'doc',
          docId: 'api/environment',
          label: 'API',
          position: 'left'
        },
        {
          type: 'doc',
          docId: 'download',
          label: 'Download',
          position: 'left'
        },
          // right
        {
          type: 'docsVersionDropdown',
          dropdownItemsAfter: [{to: '/versions', label: 'All versions'}],
          dropdownActiveClassDisabled: true,
          position: 'right',
        },
        {
          href: 'https://github.com/apache/submarine',
          label: 'GitHub',
          position: 'right',
        },
        {
          label: 'Apache',
          position: 'right',
          items: [
              {
                  label: 'Apache Software Foundation',
                  href: 'https://www.apache.org/foundation/how-it-works.html',
              },
              {
                  label: 'Events',
                  href: 'https://www.apache.org/events/current-event',
              },
              {
                  label: 'Apache License',
                  href: 'https://www.apache.org/licenses/',
              },
              {
                  label: 'Thanks',
                  href: 'https://www.apache.org/foundation/thanks.html',
              },
              {
                  label: 'Security',
                  href: 'https://www.apache.org/security/',
              },
              {
                  label: 'Sponsorship',
                  href: 'https://www.apache.org/foundation/sponsorship.html',
              },
          ],
        }
      ],
    },
    footer: {
        style: 'dark',
        logo: {
            alt: 'Apache Open Source Logo',
            src: 'https://hadoop.apache.org/asf_logo_wide.png',
            href: 'https://www.apache.org/',
        },
        links: [
            {
                title: 'Docs',
                items: [
                    {
                        label: 'Getting Started',
                        to: 'docs/',
                    },
                    {
                        label: 'API docs',
              to: 'docs/api/environment',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/apache-submarine',
            },
            {
              label: 'Slack',
              href: 'https://s.apache.org/slack-invite',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
                label: 'Blog',
                to: 'https://medium.com/@apache.submarine',
            },
              {
                  label: 'GitHub',
                  href: 'https://github.com/apache/submarine',
              },
          ],
        },
        ],
        copyright: `Apache Hadoop, Hadoop, Apache, the Apache feather logo, and the Apache Submarine project logo are
       either registered trademarks or trademarks of the Apache Software Foundation in the United States and other
        countries.<br> Copyright © ${new Date().getFullYear()} Apache Submarine is Apache2 Licensed software.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/apache/submarine/edit/master/website/',
            versions: {
              current: {
                label: `master 🏃`,
              },
            },
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  plugins: [[ require.resolve('docusaurus-lunr-search'), {
    languages: ['en', 'de'] // language codes
  }]]
};
