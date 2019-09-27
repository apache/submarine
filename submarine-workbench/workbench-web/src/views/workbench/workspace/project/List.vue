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
  <div v-if="activeTabKey === '1'">
    <a-card :bordered="false">
      <a-row>
        <a-col :sm="6" :xs="24">
          <head-info title="My Projects" content="45 Tasks" :bordered="true"/>
        </a-col>
        <a-col :sm="6" :xs="24">
          <head-info title="Average task time" content="24 minute" :bordered="true"/>
        </a-col>
        <a-col :sm="6" :xs="24">
          <head-info title="Longest task time" content="32 minute" :bordered="true"/>
        </a-col>
        <a-col :sm="6" :xs="24">
          <head-info title="Total run times" content="24"/>
        </a-col>
      </a-row>
    </a-card>

    <a-card
      style="margin-top: 24px"
      :bordered="false"
      title="My Projects">

      <div slot="extra">
        <a-radio-group>
          <a-radio-button>All</a-radio-button>
          <a-radio-button>Running</a-radio-button>
          <a-radio-button>Waiting</a-radio-button>
          <a-radio-button>Finishing</a-radio-button>
        </a-radio-group>
        <a-input-search placeholder="input project name" style="margin-left: 16px; width: 272px;" enterButton/>
      </div>

      <div class="operate">
        <a-button type="dashed" style="width: 100%" icon="plus" @click="finish('2')">Add New Project</a-button>
      </div>

      <a-list size="large" :pagination="{showSizeChanger: true, showQuickJumper: true, pageSize: 10, total: 50}">
        <a-list-item :key="index" v-for="(item, index) in data">

          <a-list-item-meta>
            <a-avatar v-if="item.mlType === 'tensorflow'" slot="avatar" size="large" shape="square" :src="'assets/tensorflow_logo.png'"/>
            <a-avatar v-if="item.mlType === 'pytorch'" slot="avatar" size="large" shape="square" :src="'assets/pytorch_logo.png'"/>

            <a slot="title" style="padding-right: 50px; color: rgba(0, 0, 0, 1);">{{ item.title }}</a>

            <a slot="title" class="list-content-item_icon"><a-icon type="star-o"/>{{ item.star }}</a>
            <a-divider slot="title" type="vertical" style="margin-left: 15px;"/>
            <a slot="title" class="list-content-item_icon"><a-icon type="like-o"/>{{ item.like }}</a>
            <a-divider slot="title" type="vertical" style="margin-left: 15px;"/>
            <a slot="title" class="list-content-item_icon"><a-icon type="message"/>{{ item.message }}</a>
            <span slot="description">
              <ellipsis :length="120">{{ item.description }}</ellipsis>
            </span>
          </a-list-item-meta>

          <div slot="actions">
            <a>编辑</a>
          </div>
          <div slot="actions">
            <a-dropdown>
              <a-menu slot="overlay">
                <a-menu-item><a>编辑</a></a-menu-item>
                <a-menu-item><a>删除</a></a-menu-item>
              </a-menu>
              <a>更多<a-icon type="down"/></a>
            </a-dropdown>
          </div>

          <div class="list-content">
            <div class="list-content-item">
              <span>Owner</span>
              <p>{{ item.owner }}</p>
            </div>
            <div class="list-content-item">
              <span>Visibility</span>
              <p>{{ item.visibility }}</p>
            </div>

            <div class="list-content-item">
              <span>Run Times</span>
              <p>{{ item.runTimes }}</p>
            </div>

            <div class="list-content-item">
              <span>Last Run</span>
              <p>{{ item.lastRunTime }}</p>
            </div>
            <div class="list-content-item">
              <a-progress :percent="item.progress.value" :status="!item.progress.status ? null : item.progress.status" style="width: 180px" />
            </div>
          </div>
        </a-list-item>
      </a-list>
      <new-project ref="newProjectModal" @ok="handleOk"/>
    </a-card>
  </div>

  <new-project v-else-if="activeTabKey === '2'">
    <a-card :bordered="true">
      <a-row>
        <a-button style="margin-left: 8px" type="primary" @click="finish('1')" icon="check">Submit</a-button>
        <a-col :sm="6" :xs="24">
          <head-info title="My Projects" content="45 Tasks" :bordered="true"/>
        </a-col>
        <a-col :sm="6" :xs="24">
          <head-info title="Average task time" content="24 minute" :bordered="true"/>
        </a-col>
        <a-col :sm="6" :xs="24">
          <head-info title="Longest task time" content="32 minute" :bordered="true"/>
        </a-col>
        <a-col :sm="6" :xs="24">
          <head-info title="Total run times" content="24"/>
        </a-col>
      </a-row>
    </a-card>
  </new-project>

</template>

<script>
import HeadInfo from '@/components/tools/HeadInfo'
import Ellipsis from '@/components/Ellipsis'
import NewProject from './NewProject'

const data = []
data.push({
  mlType: 'tensorflow',
  title: 'Tensorflow-test1',
  description: 'Basic Operations on multi-GPU (notebook) (code). A simple example to introduce multi-GPU in TensorFlow. Train a Neural Network on multi-GPU (notebook) (code)',
  owner: 'Frank',
  star: 10,
  like: 24,
  message: 2,
  visibility: 'Private',
  runTimes: 2,
  lastRunTime: '2018-07-26 22:44',
  progress: {
    value: 90
  }
})
data.push({
  mlType: 'pytorch',
  title: 'Tensorflow-test1',
  description: 'Basic Operations on multi-GPU (notebook) (code). A simple example to introduce multi-GPU in TensorFlow. Train a Neural Network on multi-GPU (notebook) (code)',
  owner: 'Frank',
  star: 10,
  like: 24,
  message: 2,
  visibility: 'Private',
  runTimes: 2,
  lastRunTime: '2018-07-26 22:44',
  progress: {
    value: 54
  }
})
data.push({
  mlType: 'tensorflow',
  title: 'Tensorflow-test1',
  description: 'Basic Operations on multi-GPU (notebook) (code). A simple example to introduce multi-GPU in TensorFlow. Train a Neural Network on multi-GPU (notebook) (code)',
  owner: 'Frank',
  star: 10,
  like: 24,
  message: 2,
  visibility: 'Public',
  runTimes: 2,
  lastRunTime: '2018-07-26 22:44',
  progress: {
    value: 66
  }
})
data.push({
  mlType: 'pytorch',
  title: 'Tensorflow-test1',
  description: 'Basic Operations on multi-GPU (notebook) (code). A simple example to introduce multi-GPU in TensorFlow. Train a Neural Network on multi-GPU (notebook) (code)',
  owner: 'Frank',
  star: 10,
  like: 24,
  message: 2,
  visibility: 'Public',
  runTimes: 2,
  lastRunTime: '2018-07-26 22:44',
  progress: {
    value: 30
  }
})
data.push({
  mlType: 'tensorflow',
  title: 'Tensorflow-test1',
  description: 'Basic Operations on multi-GPU (notebook) (code). A simple example to introduce multi-GPU in TensorFlow. Train a Neural Network on multi-GPU (notebook) (code)',
  owner: 'Frank',
  star: 10,
  like: 24,
  message: 2,
  visibility: 'Public',
  runTimes: 2,
  lastRunTime: '2018-07-26 22:44',
  progress: {
    status: 'exception',
    value: 100
  }
})

export default {
  name: 'StandardList',
  components: {
    HeadInfo,
    Ellipsis,
    NewProject
  },
  data () {
    return {
      data,
      activeTabKey: '1'
    }
  },
  onAddNewProject () {
    this.$refs.newProject.new()
  },
  methods: {
    handleOk () {
      this.$refs.table.refresh()
    },
    finish (key) {
      this.activeTabKey = key
    }
  }
}
</script>

<style lang="less" scoped>
  .ant-avatar-lg {
    width: 48px;
    height: 48px;
    line-height: 48px;
  }

  .list-content-item {
    color: rgba(0, 0, 0, 0.75);
    display: inline-block;
    vertical-align: middle;
    font-size: 14px;
    margin-left: 40px;
    span {
      line-height: 20px;
    }
    p {
      margin-top: 4px;
      margin-bottom: 0;
      line-height: 22px;
    }
  }

  .list-content-item_icon {
    color: rgba(0, 0, 0, .45);
    display: inline-block;
    vertical-align: middle;

    i {
      margin-left: 10px;
      margin-right: 10px;
    }

  }
</style>
