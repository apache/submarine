module.exports = {
  title: 'Apache Submarine',
  tagline: 'Cloud Native Machine Learning Platform',
  url: 'https://submarine.apache.org/',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'apache', // Usually your GitHub org/user name.
  projectName: 'submarine-site', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'Apache Submarine',
      logo: {
        alt: 'Apache Submarine Site Logo',
        src: 'https://github.com/apache/submarine/blob/master/docs/assets/128-black.png?raw=true',
      },
      items: [
        {
          type: 'doc',
          docId: 'gettingStarted/localDeployment',
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
              href: 'http://www.apache.org/foundation/how-it-works.html',
            },
            {
              label: 'Apache License',
              href: 'http://www.apache.org/licenses/',
            },
            {
              label: 'Sponsorship',
              href: 'http://www.apache.org/foundation/sponsorship.html',
            },
            {
              label: 'Thanks',
              href: 'http://www.apache.org/foundation/thanks.html',
            },
          ],
        }
      ],
    },
    footer: {
      style: 'dark',
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
      copyright: `Copyright © ${new Date().getFullYear()} Apache Submarine is Apache2 Licensed software.`,
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
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
