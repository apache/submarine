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
  <div>

    <div style="padding-bottom: 14px;">
      <a-button type="dashed" style="width: 100%;" icon="plus" @click="onShowNewProject()">Add New Project</a-button>
    </div>

    <div class="card-list" ref="content">
      <a-list
        :grid="{gutter: 24, lg: 3, md: 2, sm: 1, xs: 1}"
        :dataSource="data"
      >
        <a-list-item slot="renderItem" slot-scope="item">
          <template>
            <a-card :hoverable="true">
              <a-card-meta style="min-height: 148px">
                <div style="margin-bottom: 3px;" slot="title">{{ item.title }}

                </div>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.mlType === 'tensorflow'" :src="'assets/tensorflow_logo.png'"/>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.mlType === 'pytorch'" :src="'assets/pytorch_logo.png'"/>

                <div slot="title">
                  <template v-for="(tag, index) in tags">
                    <a-tooltip v-if="tag.length > 20" :key="tag" :title="tag">
                      <a-tag
                        :key="tag"
                        :closable="index !== 0"
                        :afterClose="() => handleTagClose(tag)"
                      >{{ `${tag.slice(0, 20)}...` }}</a-tag>
                    </a-tooltip>
                    <a-tag
                      v-else
                      :key="tag"
                      :closable="index !== 0"
                      :afterClose="() => handleTagClose(tag)"
                    >{{ tag }}</a-tag>
                  </template>
                  <a-input
                    v-if="tagInputVisible"
                    ref="tagInput"
                    type="text"
                    size="small"
                    :style="{ width: '78px' }"
                    :value="tagInputValue"
                    @change="handleInputChange"
                    @blur="handleTagInputConfirm"
                    @keyup.enter="handleTagInputConfirm"
                  />
                  <a-tag v-else @click="showTagInput" style="background: #fff; borderStyle: dashed;">
                    <a-icon type="plus"/>New Tag
                  </a-tag>
                </div>

                <div class="meta-content" slot="description"><ellipsis :length="140">{{ item.description }}</ellipsis><br/>
                  <a slot="description" class="list-content-item_icon"><a-icon type="star-o"/>{{ item.star }}</a>
                  <a-divider slot="title" type="vertical" class="list-content-item_divider"/>
                  <a slot="description" class="list-content-item_icon"><a-icon type="like-o"/>{{ item.like }}</a>
                  <a-divider slot="title" type="vertical" class="list-content-item_divider"/>
                  <a slot="description" class="list-content-item_icon"><a-icon type="message"/>{{ item.message }}</a>
                </div>
              </a-card-meta>
              <template class="ant-card-actions" slot="actions">
                <a><a-icon type="edit"/> Edit</a>
                <a><a-icon type="download"/> Download</a>
                <a><a-icon type="setting"/> Setting</a>
                <a>
                  <a-dropdown>
                    <a class="ant-dropdown-link" href="javascript:;">
                      <a-icon type="ellipsis"/>
                    </a>
                    <a-menu slot="overlay">
                      <a-menu-item>
                        <a href="javascript:;">1st menu item</a>
                      </a-menu-item>
                      <a-menu-item>
                        <a href="javascript:;">2nd menu item</a>
                      </a-menu-item>
                      <a-menu-item>
                        <a href="javascript:;">3rd menu item</a>
                      </a-menu-item>
                    </a-menu>
                  </a-dropdown>
                </a>
              </template>
            </a-card>
          </template>
        </a-list-item>
      </a-list>
    </div>
  </div>
</template>

<script>
import HeadInfo from '@/components/tools/HeadInfo'
import Ellipsis from '@/components/Ellipsis'
import NewProject from './NewProject'

const dataSource = []
dataSource.push(null)
for (let i = 0; i < 11; i++) {
  dataSource.push({
    title: 'Alipay',
    avatar: 'https://gw.alipayobjects.com/zos/rmsportal/WdGqmHpayyMjiEhcKoVE.png',
    description: ''
  })
}

const data = []
data.push({
  mlType: 'tensorflow',
  title: 'Tensorflow-test1',
  description: 'Basic Operations on multi-GPU (notebook) (code). A simple example to introduce multi-GPU in TensorFlow. Basic Operations on multi-GPU (notebook) (code). A simple example to introduce multi-GPU in TensorFlow. A simple example to introduce multi-GPU in TensorFlow. Train a Neural Network on multi-GPU (notebook) (code)',
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
  description: 'Basic Operations on multi-GPU (notebook) (code). ',
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
      tags: ['python', 'test', 'mnist'],
      tagInputVisible: false,
      tagInputValue: '',

      dataSource,
      data,
      activeTabKey: '1'
    }
  },

  methods: {
    onShowNewProject () {
      this.$emit('showNewProject')
    },
    handleOk () {
      this.$refs.table.refresh()
    },
    finish (key) {
      this.activeTabKey = key
    },
    handleTagClose (removeTag) {
      const tags = this.tags.filter(tag => tag !== removeTag)
      this.tags = tags
    },

    showTagInput () {
      this.tagInputVisible = true
      this.$nextTick(() => {
        this.$refs.tagInput.focus()
      })
    },

    handleInputChange (e) {
      this.tagInputValue = e.target.value
    },

    handleTagInputConfirm () {
      const inputValue = this.tagInputValue
      let tags = this.tags
      if (inputValue && !tags.includes(inputValue)) {
        tags = [...tags, inputValue]
      }

      Object.assign(this, {
        tags,
        tagInputVisible: false,
        tagInputValue: ''
      })
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
    font-size: 12px;
    color: rgba(0, 0, 0, .45);
    display: inline-block;
    vertical-align: middle;

    i {
      margin-left: 0px;
      margin-right: 10px;
      margin-top: 10px;
    }
  }

  .list-content-item_divider {
    margin-left: 15px;
    margin-right: 15px;
    margin-top: 10px;
  }

  .new-btn {
    background-color: #fff;
    border-radius: 2px;
    width: 100%;
    height: 186px;
  }
</style>
