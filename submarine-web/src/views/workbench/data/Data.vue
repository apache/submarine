<template>

  <page-view>
    <!-- actions -->
    <!--template slot="action">
      <a-button-group style="margin-right: 4px;">
        <a-button>操作</a-button>
        <a-button>操作</a-button>
        <a-button><a-icon type="ellipsis"/></a-button>
      </a-button-group>
    </template-->
    <a-row slot="extra">
      <p style="margin-left: -88px;font-size: 14px;color: rgba(0,0,0,.65)">{{ description }}</p>
    </a-row>

    <a-card :bordered="false" title="Database">
      <div slot="extra">
        <router-link :to="{ name: 'NewTableWrapper' }">
          <a-button icon="plus" type="primary" >Create Table</a-button>
        </router-link>
      </div>

      <a-row :gutter="12" :span="24">
        <a-col :span="4">
          <!--router-link :to="{ name: 'NewTableWrapper' }">
          <a-button style="width: 100%; margin-bottom: 8px" type="dashed" icon="plus" @click="newMember">
            Add Table
          </a-button></router-link-->
          <p style="font-size: 16px;">Database & Table List</p>
          <a-select
            style="width: 100%; margin-bottom: 8px"
            mode="multiple"
            placeholder="Please select database ..."
            :value="selectedItems"
            @change="onDatabaseChange"
          >
            <a-select-option v-for="item in filteredDatabases" :key="item" :value="item">
              {{ item }}
            </a-select-option>
          </a-select>
          <s-tree
            :dataSource="recentTree"
            :openKeys.sync="openKeys"
            :search="true"
            @click="handleClick"
            @add="handleAddColumn"
            @titleClick="handleTitleClick"></s-tree>
        </a-col>

        <a-col :span="20">
          <a-card
            style="margin-top: -12px"
            :bordered="false"
            :tabList="tabList"
            :activeTabKey="activeTabKey"
            @tabChange="(key) => {this.activeTabKey = key}"
          >
            <!--a-button
              v-if="activeTabKey === '1'"
              style="margin-top: -12px; margin-bottom: 8px;"
              type="primary"
              icon="plus"
              @click="handleAddColumn">Add Column</a-button-->
            <a-table
              v-if="activeTabKey === '1'"
              ref="table1"
              size="default"
              :columns="schemaColumns"
              :dataSource="schemaDataSource"
              :pagination="false"
              bordered
            >
              <template slot="title">
                <a-button @click="handleAddColumn" icon="plus">Add New Column</a-button>
              </template>
              <template v-for="col in ['name', 'type', 'comment']" :slot="col" slot-scope="text, record">
                <div :key="col">
                  <a-input
                    v-if="record.editable"
                    style="margin: -5px 0"
                    :value="text"
                    @change="e => onChangeColumn(e.target.value, record.key, col)"
                  />
                  <template v-else>{{ text }}</template>
                </div>
              </template>
              <template slot="operation" slot-scope="text, record">
                <span>
                  <span v-if="record.editable">
                    <a @click="() => onSaveColumn(record.key)">Save</a> or
                    <a-popconfirm title="Sure to cancel?" @confirm="() => onCancelColumn(record.key)">
                      <a>Cancel</a>
                    </a-popconfirm>
                  </span>
                  <span v-else>
                    <a @click="() => onEditColumn(record.key)">Edit</a>
                  </span>
                  <a-divider type="vertical" />
                  <a-popconfirm
                    v-if="schemaDataSource.length"
                    title="Sure to delete?"
                    @confirm="() => handleDelColumn(record.key)">
                    <a href="javascript:;">Delete</a>
                  </a-popconfirm>
                </span>
              </template>
            </a-table>
            <s-table
              v-if="activeTabKey === '2'"
              ref="table"
              size="default"
              :columns="tableColumns"
              :data="loadSampleData"
              bordered
            >
            </s-table>
          </a-card>
        </a-col>
      </a-row>
    </a-card>
  </page-view>
</template>

<script>
import STree from '@/components/Tree/Tree'
import { STable } from '@/components'
import { getDataTables, getSchemaColumnsData, getDatabases, getSampleData, getTableColumns } from '@/api/workbench'
import { PageView } from '@/layouts'

const databases = ['default', 'db1', 'db2']

export default {
  name: 'DataPage',
  components: {
    PageView,
    STable,
    STree
  },
  data () {
    return {
      description: 'A Submarine database is a collection of tables. A Submarine table is a collection of structured data. ',

      schemaDataSource: [],

      tabList: [
        {
          key: '1',
          tab: 'Schema'
        },
        {
          key: '2',
          tab: 'Sample Data'
        }
      ],
      activeTabKey: '1',

      selectedItems: [],

      openKeys: ['key-01'],

      // 查询参数
      queryParam: {},

      schemaColumns: [
        {
          title: 'Column Name',
          dataIndex: 'name',
          scopedSlots: { customRender: 'name' }
        },
        {
          title: 'Column Type',
          dataIndex: 'type',
          scopedSlots: { customRender: 'type' }
        },
        {
          title: 'Comment',
          dataIndex: 'comment',
          scopedSlots: { customRender: 'comment' }
        }, {
          title: 'Action',
          dataIndex: 'operation',
          scopedSlots: { customRender: 'operation' },
          width: 200
        }
      ],

      // 表头
      tableColumns: [],

      // 加载数据方法 必须为 Promise 对象
      loadSampleData: parameter => {
        return getSampleData(Object.assign(parameter, this.queryParam))
          .then(res => {
            return res.result
          })
      },

      recentTree: [],
      selectedRowKeys: [],
      selectedRows: []
    }
  },
  computed: {
    userInfo () {
      return this.$store.getters.userInfo
    },
    filteredDatabases () {
      console.log('filteredDatabases:' + this.databases)
      return databases.filter(o => !this.selectedItems.includes(o))
    }
  },

  created () {
    getDataTables().then(res => {
      console.log('getDataTables:' + res.result)
      this.recentTree = res.result
    })

    getSchemaColumnsData().then(res => {
      this.schemaDataSource = res.result
      this.cacheData = this.schemaDataSource.map(item => ({ ...item }))
    })

    this.getTableColumns()
    this.user = this.userInfo
    this.avatar = this.userInfo.avatar
  },
  methods: {
    getDatabasesMethod () {
      getDatabases().then(res => {
        console.log('getDatabases:' + res.result)
        this.databases = res.result
      })
    },

    getTableColumns () {
      getTableColumns().then(res => {
        this.tableColumns = res.result
        console.log('getTableColumns:' + this.tableColumns)
      })
    },

    newMember () {
      console.log('newMember')
    },
    handleClick (e) {
      console.log('handleClick', e)
      this.queryParam = {
        key: e.key
      }
      this.$refs.table.refresh(true)
    },
    onDatabaseChange (selectedItems) {
      this.selectedItems = selectedItems
    },
    handleAddColumn () {
      const { schemaDataSource } = this
      const newData = {
        key: 11,
        name: 'col_11',
        type: 'string',
        comment: 'comment ...',
        editable: false
      }
      this.schemaDataSource = [...schemaDataSource, newData]
    },

    onChangeColumn (value, key, column) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        target[column] = value
        this.schemaDataSource = newData
      }
    },

    onEditColumn (key) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        target.editable = true
        this.schemaDataSource = newData
      }
    },
    onSaveColumn (key) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        delete target.editable
        this.schemaDataSource = newData
        this.cacheData = newData.map(item => ({ ...item }))
      }
    },
    onCancelColumn (key) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        Object.assign(target, this.cacheData.filter(item => key === item.key)[0])
        delete target.editable
        this.schemaDataSource = newData
      }
    },

    handleDelColumn (key) {
      const schemaDataSource = [...this.schemaDataSource]
      this.schemaDataSource = schemaDataSource.filter(item => item.key !== key)
    },

    handleTitleClick (item) {
      console.log('handleTitleClick', item)
    },
    titleClick (e) {
      console.log('titleClick', e)
    },
    handleSaveOk () {

    },
    handleSaveClose () {

    },

    onSelectChange (selectedRowKeys, selectedRows) {
      this.selectedRowKeys = selectedRowKeys
      this.selectedRows = selectedRows
    }
  }
}
</script>

<style lang="less">
  .custom-tree {

    /deep/ .ant-menu-item-group-title {
      position: relative;
      &:hover {
        .btn {
          display: block;
        }
      }
    }

    /deep/ .ant-menu-item {
      &:hover {
        .btn {
          display: block;
        }
      }
    }

    /deep/ .btn {
      display: none;
      position: absolute;
      top: 0;
      right: 10px;
      width: 20px;
      height: 40px;
      line-height: 40px;
      z-index: 1050;

      &:hover {
        transform: scale(1.2);
        transition: 0.5s all;
      }
    }
  }

  .detail-layout {
    margin-left: 44px;
  }
  .text {
    color: rgba(0, 0, 0, .45);
  }

  .heading {
    color: rgba(0, 0, 0, .85);
    font-size: 20px;
  }

  .no-data {
    color: rgba(0, 0, 0, .25);
    text-align: center;
    line-height: 64px;
    font-size: 16px;

    i {
      font-size: 24px;
      margin-right: 16px;
      position: relative;
      top: 3px;
    }
  }

  .mobile {
    .detail-layout {
      margin-left: unset;
    }
    .text {

    }
    .status-list {
      text-align: left;
    }
  }

</style>
