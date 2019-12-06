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
  <a-card :bordered="false">

    <div slot="extra">
      <a-radio-group>
        <a-radio-button>All</a-radio-button>
        <a-radio-button>Owned By Me</a-radio-button>
        <a-radio-button>Accessible By Me</a-radio-button>
      </a-radio-group>
      <a-input-search style="margin-left: 16px; width: 272px;" />
      <a-button style="margin-left: 8px" type="primary" icon="plus" @click="$refs.modal.edit(null)">New Job</a-button>
    </div>

    <s-table
      ref="table"
      size="default"
      :columns="columns"
      :data="loadData"
    >
      <span slot="serial" slot-scope="text, record, index">
        {{ index + 1 }}
      </span>
      <span slot="status" slot-scope="text">
        <a-badge :status="text | statusTypeFilter" :text="text | statusFilter" />
      </span>
      <span slot="description" slot-scope="text">
        <ellipsis :length="20" tooltip>{{ text }}</ellipsis>
      </span>

      <span slot="progressRender" slot-scope="text, record">
        <div class="list-content-item">
          <a-progress :percent="record.progress.value" :status="!record.progress.status ? null : record.progress.status" style="width: 180px" />
        </div>
      </span>

      <span slot="action" slot-scope="text, record">
        <template>
          <a @click="handleEdit(record)">Start</a>
          <a-divider type="vertical" />
          <a @click="handleSub(record)">Edit</a>
        </template>
      </span>
    </s-table>
    <create-form ref="createModal" @ok="handleOk" />
    <create-job-modal ref="modal" @ok="handleOk"/>
  </a-card>
</template>

<script>
import moment from 'moment'
import { STable, Ellipsis } from '@/components'
import CreateJobModal from '@/views/workbench/job/modules/CreateJob'
import { getRoleList } from '@/api/manage'
import { getJobList } from '@/api/workbench'

const statusMap = {
  0: {
    status: 'default',
    text: 'Stop'
  },
  1: {
    status: 'processing',
    text: 'Running'
  },
  2: {
    status: 'success',
    text: 'Online'
  },
  3: {
    status: 'error',
    text: 'Exception'
  }
}

export default {
  name: 'TableList',
  components: {
    STable,
    Ellipsis,
    CreateJobModal
  },
  data () {
    return {
      description: 'A job is a way of running a notebook or on a scheduled basis.',
      logo: 'https://gw.alipayobjects.com/zos/rmsportal/nxkuOJlFJuAUhzlMTCEe.png',
      mdl: {},
      // 高级搜索 展开/关闭
      advanced: false,
      // 查询参数
      queryParam: {},
      // 表头
      columns: [
        {
          title: 'Job Name',
          dataIndex: 'no'
        },
        {
          title: 'Job ID',
          dataIndex: 'id'
        },
        {
          title: 'Owner',
          dataIndex: 'owner',
          scopedSlots: { customRender: 'description' }
        },
        {
          title: 'Actuator',
          dataIndex: 'actuator',
          scopedSlots: { customRender: 'description' }
        },
        {
          title: 'Status',
          dataIndex: 'status',
          scopedSlots: { customRender: 'status' }
        },
        {
          title: 'Progress',
          dataIndex: 'progress',
          scopedSlots: { customRender: 'progressRender' }
        },
        {
          title: 'Last Run',
          dataIndex: 'updatedAt'
        },
        {
          title: 'Action',
          dataIndex: 'action',
          width: '250px',
          scopedSlots: { customRender: 'action' }
        }
      ],
      // 加载数据方法 必须为 Promise 对象
      loadData: parameter => {
        console.log('loadData.parameter', parameter)
        return getJobList(Object.assign(parameter, this.queryParam))
          .then(res => {
            return res.result
          })
      },
      selectedRowKeys: [],
      selectedRows: [],

      // custom table alert & rowSelection
      options: {
        alert: { show: true, clear: () => { this.selectedRowKeys = [] } },
        rowSelection: {
          selectedRowKeys: this.selectedRowKeys,
          onChange: this.onSelectChange
        }
      },
      optionAlertShow: false
    }
  },
  filters: {
    statusFilter (type) {
      return statusMap[type].text
    },
    statusTypeFilter (type) {
      return statusMap[type].status
    }
  },
  created () {
    getRoleList({ t: new Date() })
  },
  methods: {
    handleEdit (record) {
      console.log(record)
      this.$refs.modal.edit(record)
    },
    handleSub (record) {
      if (record.status !== 0) {
        this.$message.info(`${record.no} 订阅成功`)
      } else {
        this.$message.error(`${record.no} 订阅失败，规则已关闭`)
      }
    },
    handleOk () {
      this.$refs.table.refresh()
    },
    onSelectChange (selectedRowKeys, selectedRows) {
      this.selectedRowKeys = selectedRowKeys
      this.selectedRows = selectedRows
    },
    toggleAdvanced () {
      this.advanced = !this.advanced
    },
    resetSearchForm () {
      this.queryParam = {
        date: moment(new Date())
      }
    }
  }
}
</script>
