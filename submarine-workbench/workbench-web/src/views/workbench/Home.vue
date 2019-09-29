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
<template>
  <div class="page-header-index-wide">
    <a-row :gutter="24">
      <a-col :sm="24" :md="12" :xl="6" :style="{ marginBottom: '24px' }">
        <chart-card :loading="loading" title="Session running" :total="2 | NumberFormat">
          <a-tooltip title="What's is session?" slot="action">
            <a-icon type="info-circle-o"/>
          </a-tooltip>
          <div>
            <mini-bar/>
          </div>
          <template slot="footer">Total: <span>120</span></template>
        </chart-card>
      </a-col>
      <a-col :sm="24" :md="12" :xl="6" :style="{ marginBottom: '24px' }">
        <chart-card :loading="loading" title="Job running" :total="1 | NumberFormat">
          <a-tooltip title="What's is job?" slot="action">
            <a-icon type="info-circle-o"/>
          </a-tooltip>
          <div>
            <mini-bar/>
          </div>
          <template slot="footer">Total: <span>100</span></template>
        </chart-card>
      </a-col>
      <a-col :sm="24" :md="12" :xl="6" :style="{ marginBottom: '24px' }">
        <chart-card :loading="loading" title="GPU / CPU" total="20%">
          <a-tooltip title="指标说明" slot="action">
            <a-icon type="info-circle-o"/>
          </a-tooltip>
          <div>
            <mini-progress color="rgb(19, 194, 194)" :target="20" :percentage="10" height="8px"/>
            <mini-progress color="rgb(19, 194, 194)" :target="50" :percentage="20" height="8px"/>
          </div>
          <template slot="footer">
            <trend flag="down" style="margin-right: 16px;">
              <span slot="term">GPU</span>
              10%
            </trend>
            <trend flag="up">
              <span slot="term">CPU</span>
              20%
            </trend>
          </template>
        </chart-card>
      </a-col>
      <a-col :sm="24" :md="12" :xl="6" :style="{ marginBottom: '24px' }">
        <chart-card :loading="loading" title="Memory(GB)" total="78%">
          <a-tooltip title="指标说明" slot="action">
            <a-icon type="info-circle-o"/>
          </a-tooltip>
          <div>
            <mini-progress color="rgb(19, 194, 194)" :target="85" :percentage="78" height="8px"/>
          </div>
          <template slot="footer">
            <trend flag="down" style="margin-right: 16px;">
              <span slot="term">Current: </span>
              160 GB
            </trend>
            <trend flag="up">
              <span slot="term">Peak: </span>
              180 GB
            </trend>
          </template>
        </chart-card>
      </a-col>
    </a-row>

    <a-row :gutter="24">
      <a-col
        style="padding: 0 12px"
        :xl="16"
        :lg="24"
        :md="24"
        :sm="24"
        :xs="24">
        <a-card :loading="loading" title="Quick Start" :bordered="false" style="margin-bottom: 24px;">
          <div class="members">
            <a-row>
              <a-col :span="6" v-for="(item, index) in quickStart" :key="index">
                <a>
                  <a-avatar size="small" :src="item.avatar"/>
                  <span class="member">{{ item.name }}</span>
                </a>
              </a-col>
            </a-row>
          </div>
        </a-card>

        <a-card
          class="project-list"
          :loading="loading"
          style="margin-bottom: 24px;"
          :bordered="false"
          title="Open Recent"
          :body-style="{ padding: 0 }">
          <a slot="extra">More</a>
          <div>
            <a-card-grid class="project-card-grid" :key="i" v-for="(item, i) in recent_projects">
              <a-card :bordered="false" :body-style="{ padding: 0 }">
                <a-card-meta>
                  <div slot="title" class="card-title">
                    <a-avatar size="small" :src="item.cover"/>
                    <a>{{ item.title }}</a>
                  </div>
                  <div slot="description" class="card-description">
                    {{ item.description }}
                  </div>
                </a-card-meta>
                <div class="project-item">
                  <a href="/#/">Tensorflow</a>
                  <span class="datetime">9小时前</span>
                </div>
              </a-card>
            </a-card-grid>
          </div>
        </a-card>
      </a-col>

      <a-col
        style="padding: 0 12px"
        :xl="8"
        :lg="24"
        :md="24"
        :sm="24"
        :xs="24">
        <a-card
          :loading="loading"
          :bordered="false"
          title="What's news?">
          <a slot="extra">More</a>
          <a-list>
            <a-list-item :key="index" v-for="(item, index) in news">
              <a-list-item-meta>
                <a-avatar slot="avatar" :src="item.user.avatar"/>
                <div slot="title">
                  <a href="#">{{ item.new.title }}</a>
                </div>
                <div slot="description">{{ item.time }}</div>
              </a-list-item-meta>
            </a-list-item>
          </a-list>
        </a-card>
      </a-col>
    </a-row>

  </div>
</template>

<script>
import { ChartCard, MiniArea, MiniBar, MiniProgress, Bar, Trend, NumberInfo, MiniSmoothArea } from '@/components'
import { mixinDevice } from '@/utils/mixin'

export default {
  name: 'Home',
  mixins: [mixinDevice],
  components: {
    ChartCard,
    MiniArea,
    MiniBar,
    MiniProgress,
    Bar,
    Trend,
    NumberInfo,
    MiniSmoothArea
  },
  data () {
    return {
      loading: true,

      news: [],

      quickStart: [
        {
          id: 1,
          name: 'New Notebook',
          avatar: 'https://gw.alipayobjects.com/zos/rmsportal/BiazfanxmamNRoxxVxka.png'
        },
        {
          id: 2,
          name: 'New Session',
          avatar: 'https://gw.alipayobjects.com/zos/rmsportal/cnrhVkzwxjPwAaCfPbdc.png'
        },
        {
          id: 1,
          name: 'New Job',
          avatar: 'https://gw.alipayobjects.com/zos/rmsportal/gaOngJwsRYRaVAuXXcmB.png'
        },
        {
          id: 1,
          name: 'New Data',
          avatar: 'https://gw.alipayobjects.com/zos/rmsportal/ubnKSIfAJTxIgXOKlciN.png'
        },
        {
          id: 1,
          name: 'New Model',
          avatar: 'https://gw.alipayobjects.com/zos/rmsportal/WhxKECPNujWoWEFNdnJE.png'
        },
        {
          id: 1,
          name: 'New Experiments',
          avatar: 'https://gw.alipayobjects.com/zos/rmsportal/WhxKECPNujWoWEFNdnJE.png'
        }
      ],

      recent_projects: []
    }
  },
  created () {
    setTimeout(() => {
      this.loading = !this.loading
    }, 1000)
  },
  mounted () {
    this.getNews()
    this.recentProjects()
  },
  methods: {
    getNews () {
      this.$http.get('/workbench/news')
        .then(res => {
          this.news = res.result
        })
    },
    recentProjects () {
      this.$http.get('/workbench/recentProjects')
        .then(res => {
          this.recent_projects = res.result
        })
    }
  }
}
</script>

<style lang="less" scoped>

  .project-list {

    .card-title {
      font-size: 0;

      a {
        color: rgba(0, 0, 0, 0.85);
        margin-left: 12px;
        line-height: 24px;
        height: 24px;
        display: inline-block;
        vertical-align: top;
        font-size: 14px;

        &:hover {
          color: #1890ff;
        }
      }
    }

    .card-description {
      color: rgba(0, 0, 0, 0.45);
      height: 44px;
      line-height: 22px;
      overflow: hidden;
    }

    .project-item {
      display: flex;
      margin-top: 8px;
      overflow: hidden;
      font-size: 12px;
      height: 20px;
      line-height: 20px;

      a {
        color: rgba(0, 0, 0, 0.45);
        display: inline-block;
        flex: 1 1 0;

        &:hover {
          color: #1890ff;
        }
      }

      .datetime {
        color: rgba(0, 0, 0, 0.25);
        flex: 0 0 auto;
        float: right;
      }
    }

    .ant-card-meta-description {
      color: rgba(0, 0, 0, 0.45);
      height: 44px;
      line-height: 22px;
      overflow: hidden;
    }
  }

  .extra-wrapper {
    line-height: 55px;
    padding-right: 24px;

    .extra-item {
      display: inline-block;
      margin-right: 24px;

      a {
        margin-left: 24px;
      }
    }
  }

  .members {
    a {
      display: block;
      margin: 12px 0;
      line-height: 24px;
      height: 24px;

      .member {
        font-size: 14px;
        color: rgba(0, 0, 0, .65);
        line-height: 24px;
        max-width: 120px;
        vertical-align: top;
        margin-left: 12px;
        transition: all 0.3s;
        display: inline-block;
      }

      &:hover {
        span {
          color: #1890ff;
        }
      }
    }
  }
</style>
