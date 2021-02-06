import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const features = [
  {
    title: 'Data Preprocessing',
    imageUrl: 'img/spark-flink.png',
    description: (
      <>
        Submarine supports data processing and algorithm development
        using spark & python through notebook
      </>
    ),
  },
  {
    title: 'Machine Learning',
    imageUrl: 'img/tf-pytorch.png',
    description: (
      <>
        Submarine supports multiple machine learning frameworks for model training.
      </>
    ),
  },
  {
    title: 'Infrastructure',
    imageUrl: 'img/yarn-k8s.png',
    description: (
      <>
        Submarine supports Yarn, Kubernetes, Docker with Resource Scheduling.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--4', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('/docs')}>
              Get Started
            </Link>
              <span className="index-ctas-github-button">
                <iframe
                    src="https://ghbtns.com/github-btn.html?user=apache&amp;repo=submarine&amp;type=star&amp;count=true&amp;size=large"
                    frameBorder={0}
                    scrolling={0}
                    width={160}
                    height={30}
                    title="GitHub Stars"
                />
              </span>
          </div>
        </div>
      </header>
      <main>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map((props, idx) => (
                  <Feature key={idx} {...props} />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
