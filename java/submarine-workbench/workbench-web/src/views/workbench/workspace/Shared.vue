<template>
  <page-view title="单号：234231029431" logo="https://gw.alipayobjects.com/zos/rmsportal/nxkuOJlFJuAUhzlMTCEe.png">
    <a-card :bordered="false">
      <a-row :gutter="8">
        <a-col :span="4">
          <s-tree
            :dataSource="recentTree"
            :openKeys.sync="openKeys"
            :search="true"
            @click="handleClick"
            @add="handleAdd"
            @titleClick="handleTitleClick"></s-tree>
        </a-col>
        <a-col :span="20">

          <s-table
            ref="table"
            size="default"
            :columns="columns"
            :data="loadData"
            bordered
          >
            <span slot="action" slot-scope="text, record">
              <template v-if="$auth('table.update')">
                <a @click="handleEdit(record)">编辑</a>
                <a-divider type="vertical" />
              </template>
              <a-dropdown>
                <a class="ant-dropdown-link">
                  更多 <a-icon type="down" />
                </a>
                <a-menu slot="overlay">
                  <a-menu-item>
                    <a href="javascript:;">详情</a>
                  </a-menu-item>
                  <a-menu-item v-if="$auth('table.disable')">
                    <a href="javascript:;">禁用</a>
                  </a-menu-item>
                  <a-menu-item v-if="$auth('table.delete')">
                    <a href="javascript:;">删除</a>
                  </a-menu-item>
                </a-menu>
              </a-dropdown>
            </span>
          </s-table>
        </a-col>
      </a-row>

    </a-card>
  </page-view>
</template>

<script>
import STree from '@/components/Tree/Tree'
import { STable } from '@/components'
import { getWorkspaceRecentFiles, getWorkspaceRecent } from '@/api/workbench'

export default {
  name: 'SharedPage',
  components: {
    STable,
    STree
  },
  data () {
    return {
      openKeys: ['key-01'],

      title: 'aaaaaaa',
      description: '段落示意：蚂蚁金服务设计平台 ant.design，用最小的工作量，无缝接入蚂蚁金服生态， 提供跨越设计与开发的体验解决方案。',

      // 查询参数
      queryParam: {},
      // 表头
      columns: [
        {
          title: 'File Name',
          dataIndex: 'description'
        },
        {
          title: 'Owner',
          dataIndex: 'owner'
        },
        {
          title: 'Status',
          dataIndex: 'status',
          needTotal: true
        },
        {
          title: 'Commit Info',
          dataIndex: 'commit',
          customRender: (text, row, index) => {
            return <a href="javascript:;">{text}</a>
          }
        },
        {
          title: 'Last Updated',
          dataIndex: 'updatedAt'
        },
        {
          title: 'Action',
          dataIndex: 'action',
          width: '150px',
          scopedSlots: { customRender: 'action' }
        }
      ],
      // 加载数据方法 必须为 Promise 对象
      loadData: parameter => {
        return getWorkspaceRecentFiles(Object.assign(parameter, this.queryParam))
          .then(res => {
            return res.result
          })
      },
      recentTree: [],
      selectedRowKeys: [],
      selectedRows: []
    }
  },
  created () {
    getWorkspaceRecent().then(res => {
      this.recentTree = res.result
    })
  },
  methods: {
    handleClick (e) {
      console.log('handleClick', e)
      this.queryParam = {
        key: e.key
      }
      this.$refs.table.refresh(true)
    },
    handleAdd (item) {
      console.log('add button, item', item)
      this.$message.info(`提示：你点了 ${item.key} - ${item.title} `)
      this.$refs.modal.add(item.key)
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
</style>
