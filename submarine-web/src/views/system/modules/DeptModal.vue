<template>
  <a-drawer
    :title="title"
    :maskClosable="true"
    :width="drawerWidth"
    placement="right"
    :closable="true"
    @close="handleCancel"
    :visible="visible"
    style="height: calc(100% - 55px);overflow: auto;padding-bottom: 53px;">

    <template slot="title">
      <div style="width: 100%;">
        <span>{{ title }}</span>
        <span style="display:inline-block;width:calc(100% - 51px);padding-right:10px;text-align: right">
          <a-button @click="toggleScreen" icon="appstore" style="height:20px;width:20px;border:0px"></a-button>
        </span>
      </div>
    </template>

    <a-spin :spinning="confirmLoading">
      <a-form :form="form">
        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Code">
          <a-input placeholder="Please entry department code" v-decorator="['deptCode', validatorRules.deptCode ]"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Name">
          <a-input placeholder="Please entry department name" v-decorator="['deptName', validatorRules.deptName ]"/>
        </a-form-item>

        <a-form-item :labelCol="labelCol" :wrapperCol="wrapperCol" label="Parent">
          <a-tree-select
            style="width:100%"
            :dropdownStyle="{maxHeight:'200px',overflow:'auto'}"
            :treeData="treeSelectData"
            showSearch
            allowClear
            treeDefaultExpandAll
            @change="onChangeParentCode"
            v-model="model.parentCode">
          </a-tree-select>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Sort">
          <a-input-number v-decorator="['sortOrder', {'initialValue':0}]"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Status"
          hasFeedback>
          <a-switch checkedChildren="available" unCheckedChildren="deleted" @change="onCheckDeleted" v-model="checkDeleted"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Description">
          <a-textarea placeholder="Please entry department description" v-decorator="['description', {'initialValue':''}]"/>
        </a-form-item>
      </a-form>
    </a-spin>

    <div class="drawer-bootom-button" v-show="!disableSubmit">
      <a-button @click="handleCancel" style="margin-right: .8rem">Cancel</a-button>
      <a-button @click="handleSubmit" type="primary" :loading="confirmLoading">Submit</a-button>
    </div>
  </a-drawer>
</template>

<script>
import pick from 'lodash.pick'
import moment from 'moment'
import { addDept, editDept, duplicateCheck, queryIdTree } from '@/api/system'

export default {
  name: 'DeptModal',
  components: {
  },
  data () {
    return {
      checkDeleted: true,
      treeSelectData: [],
      modalWidth: 800,
      drawerWidth: 700,
      modaltoggleFlag: true,
      confirmDirty: false,
      disableSubmit: false,
      validatorRules: {
        deptName: { rules: [{ required: true, message: 'Please entry department name!' }] },
        deptCode: { rules: [{ required: true, message: 'Please entry department code!' },
          { validator: this.duplicateDeptCode }, { validator: this.constraintDeptCode }] }
      },
      title: 'Operation',
      visible: false,
      model: {},
      labelCol: {
        xs: { span: 24 },
        sm: { span: 5 }
      },
      wrapperCol: {
        xs: { span: 24 },
        sm: { span: 16 }
      },
      confirmLoading: false,
      form: this.$form.createForm(this),
      url: {
        add: '/sys/dept/add',
        edit: '/sys/dept/edit'
      }
    }
  },
  created () {
  },
  computed: {
  },
  methods: {
    duplicateDeptCode (rule, value, callback) {
      var params = {
        tableName: 'sys_department',
        fieldName: 'dept_code',
        fieldVal: value,
        dataId: this.model.id
      }
      duplicateCheck(params).then((res) => {
        console.log(res)
        if (res.success) {
          callback()
        } else {
          callback(res.message)
        }
      })
    },
    constraintDeptCode (rule, value, callback) {
      console.log('value', value)
      console.log('this.model.deptCode', this.model.deptCode)
      if (this.model.deptCode === value) {
        callback()
      }
      var params = {
        tableName: 'sys_department',
        fieldName: 'parent_code',
        fieldVal: this.model.deptCode,
        dataId: this.model.id
      }
      duplicateCheck(params).then((res) => {
        console.log(res)
        if (res.success) {
          // This value not exists
          callback()
        } else {
          // This value already exists
          callback(new Error(this.model.deptCode + ' is the parent code of other departments, can not be modified!'))
        }
      })
    },
    loadTreeSelectData () {
      var that = this
      that.treeSelectData = []
      var params
      if (this.model.deptCode) {
        params = {
          disableDeptCode: this.model.deptCode
        }
      }
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
    // Window maximization switch
    toggleScreen () {
      if (this.modaltoggleFlag) {
        this.modalWidth = window.innerWidth
      } else {
        this.modalWidth = 800
      }
      this.modaltoggleFlag = !this.modaltoggleFlag
    },
    add () {
      this.edit({})
    },
    edit (record) {
      if (record.id) {
        this.checkDeleted = (record.deleted === 0)
      }
      this.resetScreenSize()
      const that = this
      that.form.resetFields()
      that.visible = true
      that.model = Object.assign({}, record)
      this.loadTreeSelectData()
      that.$nextTick(() => {
        that.form.setFieldsValue(pick(this.model, 'deptCode', 'deptName', 'parentCode', 'sortOrder', 'deleted', 'description'))
      })
    },
    close () {
      this.$emit('close')
      this.visible = false
      this.disableSubmit = false
    },
    moment,
    handleSubmit () {
      const that = this
      this.form.validateFields((err, values) => {
        if (!err) {
          that.confirmLoading = true
          const formData = Object.assign({}, values)
          formData.id = this.model.id
          formData.deleted = this.model.deleted
          formData.parentCode = this.model.parentCode
          console.log('formData', formData)

          let obj
          if (!this.model.id) {
            obj = addDept(formData)
          } else {
            obj = editDept(formData)
          }
          obj.then((res) => {
            if (res.success) {
              that.$message.success(res.message)
              that.$emit('ok')
            } else {
              that.$message.error(res.message)
            }
          }).finally(() => {
            that.confirmLoading = false
            that.close()
          })
        }
      })
    },
    handleCancel () {
      this.close()
    },
    resetScreenSize () {
      const screenWidth = document.body.clientWidth
      if (screenWidth < 500) {
        this.drawerWidth = screenWidth
      } else {
        this.drawerWidth = 700
      }
    },
    onChangeParentId (value) {
      this.model.parentCode = value
    },
    onCheckDeleted (checked) {
      this.checkDeleted = checked
      if (checked) {
        this.model.deleted = 0
      } else {
        this.model.deleted = 1
      }
    }
  }
}
</script>

<style scoped>
  .ant-table-tbody .ant-table-row td{
    padding-top:10px;
    padding-bottom:10px;
  }

  .drawer-bootom-button {
    position: absolute;
    bottom: -8px;
    width: 100%;
    border-top: 1px solid #e8e8e8;
    padding: 10px 16px;
    text-align: right;
    left: 0;
    background: #fff;
    border-radius: 0 0 2px 2px;
  }
</style>
