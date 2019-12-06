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
        :pagination="pagination"
      >
        <a-list-item slot="renderItem" slot-scope="item">
          <template>
            <a-card :hoverable="true">
              <a-card-meta style="min-height: 148px">
                <div style="margin-bottom: 3px;" slot="title">{{ item.name }}

                </div>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.type === 'PROJECT_TYPE_NOTEBOOK'" :src="'assets/notebook_logo.png'"/>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.type === 'PROJECT_TYPE_PYTHON'" :src="'assets/python_logo.png'"/>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.type === 'PROJECT_TYPE_R'" :src="'assets/r_logo.png'"/>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.type === 'PROJECT_TYPE_SCALA'" :src="'assets/scala_logo.png'"/>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.type === 'PROJECT_TYPE_TENSORFLOW'" :src="'assets/tensorflow_logo.png'"/>
                <a-avatar class="card-avatar" slot="avatar" size="large" v-if="item.type === 'PROJECT_TYPE_PYTORCH'" :src="'assets/pytorch_logo.png'"/>

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
                  <a slot="description" class="list-content-item_icon"><a-icon type="star-o"/>{{ item.starNum }}</a>
                  <a-divider slot="title" type="vertical" class="list-content-item_divider"/>
                  <a slot="description" class="list-content-item_icon"><a-icon type="like-o"/>{{ item.likeNum }}</a>
                  <a-divider slot="title" type="vertical" class="list-content-item_divider"/>
                  <a slot="description" class="list-content-item_icon"><a-icon type="message"/>{{ item.messageNum }}</a>
                </div>
              </a-card-meta>
              <template class="ant-card-actions" slot="actions">
                <a><a-icon type="edit"/> Edit</a>
                <a><a-icon type="download"/> Download</a>
                <a><a-icon type="setting"/> Setting</a>
                <a><a-icon type="delete"/> Delete</a>
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
import { queryProject } from '@/api/system'

const data = []

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
      data,
      activeTabKey: '1',
      login_user: {},
      pagination: {
        pageSize: 9,
        onChange: (page) => {
          this.pagination.current = page
          this.loadProjectData()
        }
      }
    }
  },
  computed: {
    userInfo () {
      return this.$store.getters.userInfo
    }
  },
  created () {
    this.login_user = this.userInfo
    this.loadProjectData(1)
  },
  methods: {
    loadProjectData () {
      var params = {
        userName: this.login_user.name,
        column: 'update_time',
        order: 'desc',
        pageNo: this.pagination.current,
        pageSize: this.pagination.pageSize
      }

      queryProject(params).then((res) => {
        console.log('res=', res)
        if (res.success) {
          this.data = res.result.records
          this.pagination.total = res.result.total
        } else {
          this.$message.error(res.message)
        }
      })
    },
    changePage (page, pageSize) {
      console.log('page=', page)
      console.log('pageSize=', pageSize)
    },
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
