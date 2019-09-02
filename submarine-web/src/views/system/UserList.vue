<template>
  <a-card :bordered="false">

    <div class="table-page-search-wrapper">
      <a-form layout="inline">
        <a-row :gutter="24">

          <a-col :md="6" :sm="8">
            <a-form-item label="Department">
              <a-tree-select
                style="width:100%"
                :dropdownStyle="{maxHeight:'200px',overflow:'auto'}"
                :treeData="treeSelectData"
                showSearch
                allowClear
                treeDefaultExpandAll
                v-model="queryParam.deptCode"
                @change="onChangeDeptCode">
              </a-tree-select>
            </a-form-item>
          </a-col>

          <a-col :md="6" :sm="12">
            <a-form-item label="Account Name">
              <a-input v-model="queryParam.userName"></a-input>
            </a-form-item>
          </a-col>

          <a-col :md="6" :sm="8">
            <a-form-item label="email">
              <a-input v-model="queryParam.email"></a-input>
            </a-form-item>
          </a-col>

          <a-col :md="6" :sm="8">
            <span style="float: left;overflow: hidden;" class="table-page-search-submitButtons">
              <a-button type="primary" @click="searchQuery" icon="search">Query</a-button>
              <a-button @click="handleAdd" icon="plus" style="margin-left: 8px">Add User</a-button>
            </span>
          </a-col>
        </a-row>

      </a-form>
    </div>

    <div>

      <a-table
        ref="table"
        bordered
        size="middle"
        rowKey="id"
        :columns="columns"
        :dataSource="dataSource"
        :pagination="ipagination"
        :loading="loading">

        <span slot="action" slot-scope="text, record">
          <a @click="handleEdit(record)">Edit</a>
          <a-divider type="vertical"/>

          <a-dropdown>
            <a class="ant-dropdown-link">
              More <a-icon type="down"/>
            </a>
            <a-menu slot="overlay">
              <a-menu-item>
                <a href="javascript:;" @click="handleDetail(record)">Details</a>
              </a-menu-item>

              <a-menu-item>
                <a href="javascript:;" @click="handleChangePassword(record)">Password</a>
              </a-menu-item>

              <a-menu-item>
                <a-popconfirm title="Confirm to delete?" @confirm="() => handleDelete(record.id)" okText="Ok" cancelText="Cancel">
                  <a>delete</a>
                </a-popconfirm>
              </a-menu-item>

              <a-menu-item v-if="record.status==1">
                <a-popconfirm title="Confirm to unlock?" @confirm="() => handleFrozen(record.id,2)">
                  <a>freeze</a>
                </a-popconfirm>
              </a-menu-item>

              <a-menu-item v-if="record.status==2">
                <a-popconfirm title="Confirm to lock?" @confirm="() => handleFrozen(record.id,1)">
                  <a>thaw</a>
                </a-popconfirm>
              </a-menu-item>

            </a-menu>
          </a-dropdown>
        </span>

      </a-table>
    </div>

    <user-modal ref="modalForm" @ok="modalFormOk"></user-modal>

    <password-modal ref="passwordmodal" @ok="passwordModalOk"></password-modal>
  </a-card>
</template>

<script>
import UserModal from './modules/UserModal'
import PasswordModal from './modules/PasswordModal'
import { frozenBatch, queryIdTree } from '@/api/system'
import { ListMixin } from '@/mixins/ListMixin'

export default {
  name: 'UserList',
  mixins: [ListMixin],
  components: {
    UserModal,
    PasswordModal
  },
  data () {
    return {
      treeSelectData: [],
      selectedRowKeys: [],
      description: 'You can check the user, delete the user, lock and unlock the user, etc.',
      queryParam: {},
      columns: [
        {
          title: 'Account Name',
          align: 'center',
          width: 150,
          dataIndex: 'userName'
        },
        {
          title: 'Real Name',
          align: 'center',
          width: 150,
          dataIndex: 'realName'
        },
        {
          title: 'Department',
          align: 'center',
          width: 150,
          dataIndex: 'deptName'
        },
        {
          title: 'Role',
          align: 'center',
          width: 150,
          dataIndex: 'roleCode'
        },
        {
          title: 'Status',
          align: 'center',
          width: 120,
          dataIndex: 'status@dict'
        },
        {
          title: 'Sex',
          align: 'center',
          width: 120,
          dataIndex: 'sex@dict'
        },
        {
          title: 'Email',
          align: 'center',
          width: 120,
          dataIndex: 'email'
        },
        {
          title: 'Create Time',
          align: 'center',
          width: 180,
          dataIndex: 'createTime'
        },
        {
          title: 'Action',
          dataIndex: 'action',
          scopedSlots: { customRender: 'action' },
          align: 'center',
          width: 120
        }
      ],
      url: {
        list: '/sys/user/list',
        delete: '/sys/user/delete',
        deleteBatch: '/sys/user/deleteBatch'
      }
    }
  },
  created () {
    this.loadTreeSelectData()
  },
  computed: {
    importExcelUrl: function () {
      // return `${window._CONFIG['domianURL']}/${this.url.importExcelUrl}`
      return null
    }
  },
  methods: {
    loadTreeSelectData () {
      var that = this
      that.treeSelectData = []
      var params
      console.log('params', params)
      queryIdTree(params).then((res) => {
        if (res.success) {
          console.log('loadTreeSelectData:', res.result)
          for (let i = 0; i < res.result.length; i++) {
            const temp = res.result[i]
            that.treeSelectData.push(temp)
          }
        }
      })
    },
    batchFrozen: function (status) {
      if (this.selectedRowKeys.length <= 0) {
        this.$message.warning('Please select a record！')
        return false
      } else {
        let ids = ''
        const that = this
        that.selectedRowKeys.forEach(function (val) {
          ids += val + ','
        })
        that.$confirm({
          title: 'Confirmation operation',
          content: (status === 1 ? 'Unlock' : 'Lock') + 'selected account?',
          onOk: function () {
            frozenBatch({ ids: ids, status: status }).then((res) => {
              if (res.success) {
                that.$message.success(res.message)
                that.loadData()
              } else {
                that.$message.warning(res.message)
              }
            })
          }
        })
      }
    },
    handleMenuClick (e) {
      if (e.key === 1) {
        this.batchDel()
      } else if (e.key === 2) {
        this.batchFrozen(2)
      } else if (e.key === 3) {
        this.batchFrozen(1)
      }
    },
    handleFrozen: function (id, status) {
      const that = this
      frozenBatch({ ids: id, status: status }).then((res) => {
        if (res.success) {
          that.$message.success(res.message)
          that.loadData()
        } else {
          that.$message.warning(res.message)
        }
      })
    },
    handleChangePassword (record) {
      this.$refs.passwordmodal.show(record.id, record.userName)
    },
    passwordModalOk () {

    }
  }

}
</script>
