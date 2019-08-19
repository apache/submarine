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
          label="Data value">
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
          The smaller the value, the higher the front, the support for decimals
        </a-form-item>

        <a-form-item
          :labelCol="labelCol"
          :wrapperCol="wrapperCol"
          label="Whether to enable"
          hasFeedback>
          <a-switch checkedChildren="Enable" unCheckedChildren="Disable" @change="onChose" v-model="visibleCheck"/>
        </a-form-item>

      </a-form>
    </a-spin>
  </a-modal>
</template>

<script>
import pick from 'lodash.pick'
import { addDictItem, editDictItem } from '@/api/system'

export default {
  name: 'DictItemModal',
  data () {
    return {
      title: 'Operation',
      visible: false,
      visibleCheck: true,
      model: {},
      dictId: '',
      status: 1,
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
        itemText: { rules: [{ required: true, message: 'Please entry name!' }] },
        itemValue: { rules: [{ required: true, message: 'Please entry data value!' }] }
      }
    }
  },
  created () {
  },
  methods: {
    add (dictId) {
      this.dictId = dictId
      this.edit({})
    },
    edit (record) {
      if (record.id) {
        this.dictId = record.dictId
        this.visibleCheck = (record.status === 1)
      }
      this.form.resetFields()
      this.model = Object.assign({}, record)
      this.model.dictId = this.dictId
      this.model.status = this.status
      this.visible = true
      this.$nextTick(() => {
        this.form.setFieldsValue(pick(this.model, 'itemText', 'itemValue', 'description', 'sortOrder'))
      })
    },
    onChose (checked) {
      if (checked) {
        this.status = 1
        this.visibleCheck = true
      } else {
        this.status = 0
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
          formData.status = this.status
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
