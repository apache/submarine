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
          label="Name">
          <a-input placeholder="Please entry name" v-decorator="['itemText', validatorRules.itemText]"/>
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Value">
          <a-input placeholder="Please entry data value" v-decorator="['itemValue', validatorRules.itemValue]"/>
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
          label="sort value">
          <a-input-number :min="1" v-decorator="['sortOrder',{'initialValue':1}]"/>
          &nbsp;Sorting in the list
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
      dictId: '',
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
        itemText: { rules: [{ required: true, message: 'Please entry name!' },
          { validator: this.validateItemText }] },
        itemValue: { rules: [{ required: true, message: 'Please entry data value!' },
          { validator: this.validateItemValue }] }
      }
    }
  },
  created () {
  },
  methods: {
    validateItemText (rule, value, callback) {
      // 重复校验
      var params = {
        tableName: 'sys_dict_item',
        fieldName: 'item_text',
        fieldVal: value,
        equalFieldName: 'dict_id',
        equalFieldVal: this.dictId,
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
    validateItemValue (rule, value, callback) {
      // 重复校验
      var params = {
        tableName: 'sys_dict_item',
        fieldName: 'item_value',
        fieldVal: value,
        equalFieldName: 'dict_id',
        equalFieldVal: this.dictId,
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
    add (dictId) {
      this.dictId = dictId
      this.edit({})
    },
    edit (record) {
      if (record.id) {
        this.dictId = record.dictId
        this.visibleCheck = (record.deleted === 0)
      }
      this.form.resetFields()
      this.model = Object.assign({}, record)
      this.model.dictId = this.dictId
      this.model.deleted = this.deleted
      this.visible = true
      this.$nextTick(() => {
        this.form.setFieldsValue(pick(this.model, 'itemText', 'itemValue', 'description', 'sortOrder'))
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
          values.itemText = (values.itemText || '').trim()
          values.itemValue = (values.itemValue || '').trim()
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
