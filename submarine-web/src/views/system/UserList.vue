<template>
  <a-card :bordered="false">

    <!-- 查询区域 -->
    <div class="table-page-search-wrapper">
      <a-form layout="inline">
        <a-row :gutter="24">

          <a-col :md="6" :sm="12">
            <a-form-item label="User Name">
              <a-input></a-input>
            </a-form-item>
          </a-col>

          <a-col :md="6" :sm="8">
            <a-form-item label="email">
              <a-input v-model="queryParam.email"></a-input>
            </a-form-item>
          </a-col>

          <a-col :md="6" :sm="8">
            <a-form-item label="Status">
              <a-select v-model="queryParam.sex">
                <a-select-option value="">All status</a-select-option>
                <a-select-option value="1">lock</a-select-option>
                <a-select-option value="2">unlock</a-select-option>
              </a-select>
            </a-form-item>
          </a-col>

          <a-col :md="6" :sm="8">
            <span style="float: left;overflow: hidden;" class="table-page-search-submitButtons">
              <a-button type="primary" icon="search">Query</a-button>
              <a-button type="primary" icon="reload" style="margin-left: 8px">Reset</a-button>
            </span>
          </a-col>

        </a-row>
      </a-form>
    </div>

    <!-- 操作按钮区域 -->
    <div class="table-operator" style="border-top: 5px">
      <a-button @click="handleAdd" type="primary" icon="plus">Add User</a-button>
      <a-button type="primary" icon="download" @click="handleExportXls('User Information')">Export</a-button>
      <a-upload
        name="file"
        :showUploadList="false"
        :multiple="false"
        :headers="tokenHeader"
        :action="importExcelUrl"
        @change="handleImportExcel">
        <a-button type="primary" icon="import">Import Excel</a-button>
      </a-upload>
      <a-dropdown v-if="selectedRowKeys.length > 0">
        <a-menu slot="overlay" @click="handleMenuClick">
          <a-menu-item key="1">
            <a-icon type="delete" @click="batchDel"/>
            Delete
          </a-menu-item>
          <a-menu-item key="2">
            <a-icon type="lock" @click="batchFrozen('2')"/>
            Frozen
          </a-menu-item>
          <a-menu-item key="3">
            <a-icon type="unlock" @click="batchFrozen('1')"/>
            thaw
          </a-menu-item>
        </a-menu>
        <a-button style="margin-left: 8px">
          batch operation
          <a-icon type="down"/>
        </a-button>
      </a-dropdown>
    </div>

    <!-- table区域-begin -->
    <div>
      <div class="ant-alert ant-alert-info" style="margin-bottom: 16px;">
        <i class="anticon anticon-info-circle ant-alert-icon"></i>Selectd&nbsp;<a style="font-weight: 600">{{ selectedRowKeys.length }}</a> items&nbsp;&nbsp;
        <a style="margin-left: 24px" @click="onClearSelected">Clean selected</a>
      </div>

      <a-table
        ref="table"
        bordered
        size="middle"
        rowKey="id"
        :columns="columns"
        :dataSource="dataSource"
        :pagination="ipagination"
        :loading="loading"
        :rowSelection="{selectedRowKeys: selectedRowKeys, onChange: onSelectChange}"
        @change="handleTableChange">

        <template slot="avatarslot" slot-scope="text, record">
          <div class="anty-img-wrap">
            <a-avatar shape="square" :src="getAvatarView(record.avatar)" icon="user"/>
          </div>
        </template>

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
                <a href="javascript:;" @click="handleChangePassword(record.username)">Password</a>
              </a-menu-item>

              <a-menu-item>
                <a-popconfirm title="Confirm to delete?" @confirm="() => handleDelete(record.id)">
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
    <!-- table区域-end -->

    <user-modal ref="modalForm" @ok="modalFormOk"></user-modal>

    <password-modal ref="passwordmodal" @ok="passwordModalOk"></password-modal>
  </a-card>
</template>

<script>
import UserModal from './modules/UserModal'
import PasswordModal from './modules/PasswordModal'
import { frozenBatch } from '@/api/system'
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
      selectedRowKeys: [],
      description: 'You can check the user, delete the user, lock and unlock the user, etc.',
      queryParam: {},
      columns: [
        /* {
            title: '#',
            dataIndex: '',
            key:'rowIndex',
            width:60,
            align:"center",
            customRender:function (t,r,index) {
              return parseInt(index)+1;
            }
          }, */
        {
          title: 'Account Name',
          align: 'center',
          dataIndex: 'username',
          width: 120
        },
        {
          title: 'User Name',
          align: 'center',
          width: 120,
          dataIndex: 'realname'
        },
        {
          title: 'Department',
          align: 'center',
          width: 120,
          dataIndex: 'department'
        },
        {
          title: 'Phone',
          align: 'center',
          width: 120,
          dataIndex: 'phone'
        },
        {
          title: 'Email',
          align: 'center',
          dataIndex: 'email'
        },
        {
          title: 'Status',
          align: 'center',
          width: 120,
          dataIndex: 'status_dictText'
        },
        {
          title: 'Create Time',
          align: 'center',
          width: 150,
          dataIndex: 'createTime'
        },
        {
          title: 'Action',
          dataIndex: 'action',
          scopedSlots: { customRender: 'action' },
          align: 'center',
          width: 170
        }

      ],
      url: {
        list: '/user/list',
        delete: '/user/delete',
        deleteBatch: '/user/deleteBatch'
      }
    }
  },
  computed: {
    importExcelUrl: function () {
      // return `${window._CONFIG['domianURL']}/${this.url.importExcelUrl}`
      return null
    }
  },
  methods: {
    getAvatarView: function (avatar) {
      return this.url.imgerver + '/' + avatar
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
                that.onClearSelected()
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
    handleChangePassword (username) {
      this.$refs.passwordmodal.show(username)
    },
    handleAgentSettings (username) {

    },
    passwordModalOk () {

    }
  }

}
</script>
