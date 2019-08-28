<template>
  <a-card :bordered="false">

    <!-- 左侧面板 -->
    <div class="table-page-search-wrapper">
      <a-form layout="inline">
        <a-row :gutter="12">
          <a-col :md="7" :sm="8">
            <a-form-item label="Code" :labelCol="labelCol" :wrapperCol="wrapperCol">
              <a-input placeholder="Please entry department code" v-model="queryParam.deptCode"></a-input>
            </a-form-item>
          </a-col>
          <a-col :md="7" :sm="8">
            <a-form-item label="Name" :labelCol="labelCol" :wrapperCol="wrapperCol">
              <a-input placeholder="Please entry department name" v-model="queryParam.deptName"></a-input>
            </a-form-item>
          </a-col>
          <a-col :md="7" :sm="8">
            <span style="float: left;overflow: hidden;" class="table-page-search-submitButtons">
              <a-button type="primary" @click="searchQuery" icon="search">Query</a-button>
              <a-button @click="handleAdd" icon="plus" style="margin-left: 8px">Add</a-button>
            </span>
          </a-col>
        </a-row>
      </a-form>

      <div style="background: #fff; padding-left:16px; padding-bottom: 8px; height: 100%;">
        <a-alert v-if="this.responseAttributes.showAlert===true" type="error" :showIcon="true">
          <div slot="message">
            The department's level is set incorrectly. Now show all department in a list.
            Click
            <a-popconfirm title="Confirm reset？" @confirm="onResetParentDept" okText="Reset" cancelText="Cancel">
              <a style="font-weight:bold"> RESET </a>
            </a-popconfirm>
            to clear the level settings for all department.
          </div>
        </a-alert>
        <a-table
          ref="table"
          rowKey="id"
          size="middle"
          :columns="columns"
          :dataSource="dataSource"
          :pagination="false"
          :loading="loading"
          @change="handleTableChange">
          <span slot="deleted" slot-scope="text">
            <a-tag v-if="text==0" color="blue">available</a-tag>
            <a-tag v-if="text==1" color="red">deleted</a-tag>
          </span>
          <span slot="action" slot-scope="text, record">
            <a @click="handleEdit(record)">
              <a-icon type="edit"/>
              Edit
            </a>
            <a-divider type="vertical"/>
            <a-popconfirm v-if="record.deleted==1" title="Confirm restore?" @confirm="() =>handleDelete(record.id, 0)" okText="Yes" cancelText="No">
              <a>Restore</a>
            </a-popconfirm>
            <a-popconfirm v-else title="Confirm delete?" @confirm="() =>handleDelete(record.id, 1)" okText="Yes" cancelText="No">
              <a>Delete</a>
            </a-popconfirm>
          </span>
        </a-table>
      </div>
    </div>
    <dept-modal ref="modalForm" @ok="modalFormOk"></dept-modal>
  </a-card>
</template>

<script>
import { filterObj } from '@/utils/util'
import { ListMixin } from '@/mixins/ListMixin'
import { resetParentDept } from '@/api/system'
import DeptModal from './modules/DeptModal'

export default {
  name: 'DeptList',
  mixins: [ListMixin],
  components: { DeptModal },
  data () {
    return {
      description: 'System Department Manager',
      visible: false,
      queryParam: {
        deptCode: '',
        deptName: ''
      },
      // Table Header
      columns: [
        {
          title: '',
          width: 100,
          align: 'center'
        },
        {
          title: 'Code',
          align: 'left',
          dataIndex: 'deptCode',
          key: 'deptCode'
        },
        {
          title: 'Name',
          align: 'left',
          dataIndex: 'deptName',
          key: 'deptName'
        },
        {
          title: 'Parent Deptartment',
          align: 'left',
          dataIndex: 'parentName',
          key: 'parentName'
        },
        {
          title: 'Description',
          align: 'left',
          dataIndex: 'description',
          key: 'description'
        },
        {
          title: 'Status',
          align: 'left',
          dataIndex: 'deleted',
          key: 'deleted',
          scopedSlots: { customRender: 'deleted' }
        },
        {
          title: 'Action',
          dataIndex: 'action',
          align: 'center',
          scopedSlots: { customRender: 'action' }
        }
      ],
      labelCol: {
        xs: { span: 8 },
        sm: { span: 5 }
      },
      wrapperCol: {
        xs: { span: 16 },
        sm: { span: 19 }
      },
      url: {
        list: '/sys/dept/tree',
        delete: '/sys/dept/delete',
        deleteBatch: '/sys/dept/deleteBatch'
      }
    }
  },
  methods: {
    getQueryParams () {
      var param = Object.assign({}, this.queryParam, this.isorter)
      param.field = this.getQueryField()
      return filterObj(param)
    },
    // 重置字典类型搜索框的内容
    searchReset () {
      var that = this
      that.queryParam.deptName = ''
      that.queryParam.deptCode = ''
      that.loadData()
    },
    onResetParentDept () {
      var that = this
      that.confirmLoading = true
      const obj = resetParentDept()
      obj.then((res) => {
        if (res.success) {
          that.$message.success(res.message)
          that.$emit('ok')
          that.loadData()
        } else {
          that.$message.error(res.message)
        }
      }).finally(() => {
        that.confirmLoading = false
      })
    }
  },
  watch: {
    openKeys (val) {
      console.log('openKeys', val)
    }
  }
}
</script>
