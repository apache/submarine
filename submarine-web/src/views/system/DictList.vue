<template>
  <a-card :bordered="false">

    <!-- 左侧面板 -->
    <div class="table-page-search-wrapper">
      <a-form layout="inline">
        <a-row :gutter="12">
          <a-col :md="7" :sm="8">
            <a-form-item label="Dict Name" :labelCol="{span: 4}" :wrapperCol="{span: 14, offset: 1}">
              <a-input placeholder="Please entry dict name" v-model="queryParam.dictName"></a-input>
            </a-form-item>
          </a-col>
          <a-col :md="7" :sm="8">
            <a-form-item label="Dict Code" :labelCol="{span: 4}" :wrapperCol="{span: 14, offset: 1}">
              <a-input placeholder="Please entry dict code" v-model="queryParam.dictCode"></a-input>
            </a-form-item>
          </a-col>
          <a-col :md="7" :sm="8">
            <span style="float: left;overflow: hidden;" class="table-page-search-submitButtons">
              <a-button type="primary" @click="searchQuery" icon="search">Query</a-button>
              <a-button type="primary" @click="handleAdd" icon="plus" style="margin-left: 8px">Add</a-button>
            </span>
          </a-col>
        </a-row>
      </a-form>

      <a-table
        ref="table"
        rowKey="id"
        size="middle"
        :columns="columns"
        :dataSource="dataSource"
        :pagination="ipagination"
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
          <a @click="editDictItem(record)"><a-icon type="setting"/> Configuration</a>
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
    <dict-modal ref="modalForm" @ok="modalFormOk"></dict-modal>  <!-- 字典类型 -->
    <dict-item-list ref="dictItemList"></dict-item-list>
  </a-card>
</template>

<script>
import { filterObj } from '@/utils/util'
import { ListMixin } from '@/mixins/ListMixin'
import DictModal from './modules/DictModal'
import DictItemList from './DictItemList'

export default {
  name: 'DictList',
  mixins: [ListMixin],
  components: { DictModal, DictItemList },
  data () {
    return {
      description: 'System Dict Manager',
      visible: false,
      // 查询条件
      queryParam: {
        dictCode: '',
        dictName: ''
      },
      // 表头
      columns: [
        {
          title: '#',
          dataIndex: '',
          key: 'rowIndex',
          width: 120,
          align: 'center',
          customRender: function (t, r, index) {
            return parseInt(index) + 1
          }
        },
        {
          title: 'Dict Code',
          align: 'left',
          dataIndex: 'dictCode'
        },
        {
          title: 'Dict Name',
          align: 'left',
          dataIndex: 'dictName'
        },
        {
          title: 'Description',
          align: 'left',
          dataIndex: 'description'
        },
        {
          title: 'Status',
          align: 'left',
          dataIndex: 'deleted',
          scopedSlots: { customRender: 'deleted' }
        },
        {
          title: 'Action',
          dataIndex: 'action',
          align: 'center',
          scopedSlots: { customRender: 'action' }
        }
      ],
      dict: '',
      labelCol: {
        xs: { span: 8 },
        sm: { span: 5 }
      },
      wrapperCol: {
        xs: { span: 16 },
        sm: { span: 19 }
      },
      url: {
        list: '/sys/dict/list',
        delete: '/sys/dict/delete',
        deleteBatch: '/sys/dict/deleteBatch'
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
    getQueryParams () {
      var param = Object.assign({}, this.queryParam, this.isorter)
      param.field = this.getQueryField()
      param.pageNo = this.ipagination.current
      param.pageSize = this.ipagination.pageSize
      return filterObj(param)
    },
    // 取消选择
    cancelDict () {
      this.dict = ''
      this.visible = false
      this.loadData()
    },
    // 编辑字典数据
    editDictItem (record) {
      this.$refs.dictItemList.edit(record)
    },
    // 重置字典类型搜索框的内容
    searchReset () {
      var that = this
      that.queryParam.dictName = ''
      that.queryParam.dictCode = ''
      that.loadData(this.ipagination.current)
    }
  },
  watch: {
    openKeys (val) {
      console.log('openKeys', val)
    }
  }
}
</script>
