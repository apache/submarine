<template>
  <a-card :bordered="false">

    <a-drawer
      title="Dict Item List"
      :width="screenWidth"
      @close="onClose"
      :visible="visible"
    >

      <div
        :style="{
          padding:'10px',
          border: '1px solid #e9e9e9',
          background: '#fff',
        }">

        <div class="table-page-search-wrapper">
          <a-form layout="inline" :form="form">
            <a-row :gutter="10">
              <a-col :md="8" :sm="12">
                <a-form-item label="Code">
                  <a-input style="width: 120px;" placeholder="Dict name" v-model="queryParam.itemCode"></a-input>
                </a-form-item>
              </a-col>
              <a-col :md="8" :sm="12">
                <a-form-item label="Name">
                  <a-input style="width: 120px;" placeholder="Dict value" v-model="queryParam.itemName"></a-input>
                </a-form-item>
              </a-col>
              <a-col :md="8" :sm="24">
                <span style="float: left;" class="table-page-search-submitButtons">
                  <a-button type="primary" @click="searchQuery">Query</a-button>
                  <a-button @click="handleAdd" style="margin-left: 8px" icon="plus">New</a-button>
                </span>
              </a-col>
            </a-row>
          </a-form>
        </div>
        <div>
          <a-table
            ref="table"
            rowKey="id"
            size="middle"
            :columns="columns"
            :dataSource="dataSource"
            :pagination="ipagination"
            :loading="loading"
            @change="handleTableChange"
          >
            <span slot="deleted" slot-scope="text">
              <a-tag v-if="text==0" color="blue">available</a-tag>
              <a-tag v-if="text==1" color="red">deleted</a-tag>
            </span>
            <span slot="action" slot-scope="text, record">
              <a @click="handleEdit(record)"><a-icon type="edit"/> Edit</a>
              <!-- delete/restore menu
              <a-divider type="vertical"/>
              <a-popconfirm v-if="record.deleted==1" title="Confirm restore?" @confirm="() =>handleDelete(record.id, 0)" okText="Yes" cancelText="No">
                <a>Restore</a>
              </a-popconfirm>
              <a-popconfirm v-else title="Confirm delete?" @confirm="() =>handleDelete(record.id, 1)" okText="Yes" cancelText="No">
                <a>Delete</a>
              </a-popconfirm>
              -->
            </span>
            <span slot="description" slot-scope="text">
              <ellipsis :length="14" tooltip>{{ text }}</ellipsis>
            </span>
          </a-table>
        </div>
      </div>
    </a-drawer>
    <dict-item-modal ref="modalForm" @ok="modalFormOk"></dict-item-modal>
  </a-card>
</template>

<script>
import pick from 'lodash.pick'
import { filterObj } from '@/utils/util'
import { ListMixin } from '@/mixins/ListMixin'
import { Ellipsis } from '@/components'
import DictItemModal from './modules/DictItemModal'

export default {
  name: 'DictItemList',
  mixins: [ListMixin],
  components: { Ellipsis, DictItemModal },
  data () {
    return {
      columns: [
        {
          title: 'Code',
          align: 'center',
          dataIndex: 'itemCode',
          scopedSlots: { customRender: 'description' }
        },
        {
          title: 'Name',
          align: 'center',
          dataIndex: 'itemName',
          scopedSlots: { customRender: 'description' }
        },
        {
          title: 'Status',
          align: 'center',
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
      queryParam: {
        dictCode: '',
        dictName: '',
        itemCode: '',
        deleted: '1'
      },
      title: 'action',
      visible: false,
      screenWidth: 800,
      model: {},
      dictCode: '',
      deleted: 1,
      labelCol: {
        xs: { span: 5 },
        sm: { span: 5 }
      },
      wrapperCol: {
        xs: { span: 12 },
        sm: { span: 12 }
      },
      form: this.$form.createForm(this),
      validatorRules: {
        itemCode: { rules: [{ required: true, message: 'Please entry code!' }] },
        itemName: { rules: [{ required: true, message: 'Please entry name!' }] }
      },
      url: {
        list: '/sys/dictItem/list',
        delete: '/sys/dictItem/delete',
        deleteBatch: '/sys/dictItem/deleteBatch'
      }
    }
  },
  created () {
    // 当页面初始化时,根据屏幕大小来给抽屉设置宽度
    this.resetScreenSize()
  },
  methods: {
    add (dictCode) {
      this.dictCode = dictCode
      this.edit({})
    },
    edit (record) {
      if (record.dictCode) {
        this.dictCode = record.dictCode
      }
      this.queryParam = {}
      this.form.resetFields()
      this.model = Object.assign({}, record)
      this.model.dictCode = this.dictCode
      this.model.deleted = this.deleted
      this.visible = true
      this.$nextTick(() => {
        this.form.setFieldsValue(pick(this.model, 'itemCode', 'itemName'))
      })
      // 当其它模块调用该模块时,调用此方法加载字典数据
      this.loadData()
    },

    getQueryParams () {
      var param = Object.assign({}, this.queryParam)
      param.dictCode = this.dictCode
      param.field = this.getQueryField()
      param.pageNo = this.ipagination.current
      param.pageSize = this.ipagination.pageSize
      return filterObj(param)
    },

    // 添加字典数据
    handleAdd () {
      this.$refs.modalForm.add(this.dictCode)
      this.$refs.modalForm.title = 'New'
    },
    showDrawer () {
      this.visible = true
    },
    onClose () {
      this.visible = false
      this.form.resetFields()
      this.dataSource = []
    },
    // 抽屉的宽度随着屏幕大小来改变
    resetScreenSize () {
      const screenWidth = document.body.clientWidth
      if (screenWidth < 600) {
        this.screenWidth = screenWidth
      } else {
        this.screenWidth = 600
      }
    }
  }
}
</script>
