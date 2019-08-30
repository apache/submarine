<template>
  <a-modal
    :title="title"
    :width="800"
    :visible="visible"
    :confirmLoading="confirmLoading"
    @ok="handleOk"
    @cancel="handleCancel"
    cancelText="Close"
  >
    <a-spin :spinning="confirmLoading">
      <a-form :form="form">

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Code">
          <a-input placeholder="Please entry code" v-decorator="['itemCode', validatorRules.itemCode]"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Name">
          <a-input placeholder="Please entry name" v-decorator="['itemName', validatorRules.itemName]"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="description">
          <a-input v-decorator="['description']"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="sort order">
          <a-input-number :min="1" v-decorator="['sortOrder',{'initialValue':1}]"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Status"
          hasFeedback>
          <a-switch checkedChildren="available" unCheckedChildren="deleted" @change="onChose" v-model="visibleCheck"/>
        </a-form-item>

      </a-form>
    </a-spin>
  </a-modal>
</template>

<script>
import pick from 'lodash.pick'
import { addDictItem, editDictItem, duplicateCheck } from '@/api/system'

export default {
  name: 'DictItemModal',
  data () {
    return {
      title: 'Operation',
      visible: false,
      visibleCheck: true,
      model: {},
      dictCode: '',
      deleted: 0,
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
      validatorRules: {
        itemCode: { rules: [{ required: true, message: 'Please entry name!' },
          { validator: this.validateItemCode }] },
        itemName: { rules: [{ required: true, message: 'Please entry data value!' },
          { validator: this.validateItemName }] }
      }
    }
  },
  created () {
  },
  methods: {
    validateItemCode (rule, value, callback) {
      // Check if the dict item code is duplicated
      var params = {
        tableName: 'sys_dict_item',
        fieldName: 'item_code',
        fieldVal: value,
        equalFieldName: 'dict_code',
        equalFieldVal: this.dictCode,
        dataId: this.model.id
      }
      duplicateCheck(params).then((res) => {
        if (res.success) {
          callback()
        } else {
          callback(res.message)
        }
      })
    },
    validateItemName (rule, value, callback) {
      // Check if the dict item name is duplicated
      var params = {
        tableName: 'sys_dict_item',
        fieldName: 'item_name',
        fieldVal: value,
        equalFieldName: 'dict_code',
        equalFieldVal: this.dictCode,
        dataId: this.model.id
      }
      duplicateCheck(params).then((res) => {
        if (res.success) {
          callback()
        } else {
          callback(res.message)
        }
      })
    },
    add (dictCode) {
      this.dictCode = dictCode
      this.edit({})
    },
    edit (record) {
      if (record.id) {
        this.dictCode = record.dictCode
        this.visibleCheck = (record.deleted === 0)
      }
      this.form.resetFields()
      this.model = Object.assign({}, record)
      this.model.dictCode = this.dictCode
      this.model.deleted = this.deleted
      this.visible = true
      this.$nextTick(() => {
        this.form.setFieldsValue(pick(this.model, 'itemCode', 'itemName', 'description', 'sortOrder'))
      })
    },
    onChose (checked) {
      if (checked) {
        this.deleted = 0
        this.visibleCheck = true
      } else {
        this.deleted = 1
        this.visibleCheck = false
      }
    },
    // 确定
    handleOk () {
      const that = this
      // 触发表单验证
      this.form.validateFields((err, values) => {
        if (!err) {
          that.confirmLoading = true
          values.itemCode = (values.itemCode || '').trim()
          values.itemName = (values.itemName || '').trim()
          values.description = (values.description || '').trim()
          const formData = Object.assign(this.model, values)
          formData.deleted = this.deleted
          let obj
          if (!this.model.id) {
            obj = addDictItem(formData)
          } else {
            obj = editDictItem(formData)
          }
          obj.then((res) => {
            if (res.success) {
              that.$message.success(res.message)
              that.$emit('ok')
            } else {
              that.$message.warning(res.message)
            }
          }).finally(() => {
            that.confirmLoading = false
            that.close()
          })
        }
      })
    },
    // 关闭
    handleCancel () {
      this.close()
    },
    close () {
      this.$emit('close')
      this.visible = false
    }
  }
}
</script>
