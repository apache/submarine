<template>
  <a-modal
    title="Reset Password"
    :width="800"
    :visible="visible"
    :confirmLoading="confirmLoading"
    @ok="handleSubmit"
    @cancel="handleCancel"
    cancelText="Close"
    okText="Ok"
    style="top:20px;"
  >
    <a-spin :spinning="confirmLoading">
      <a-form :form="form">

        <a-form-item label="Account Name" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-input placeholder="Please input account name" v-decorator="[ 'userName', {}]" :readOnly="true"/>
        </a-form-item>

        <a-form-item label="Password" :labelCol="labelCol" :wrapperCol="wrapperCol" hasFeedback >
          <a-input type="password" placeholder="Please input password" v-decorator="[ 'password', validatorRules.password]" />
        </a-form-item>

        <a-form-item label="Confirm password" :labelCol="labelCol" :wrapperCol="wrapperCol" hasFeedback >
          <a-input type="password" @blur="handleConfirmBlur" placeholder="Pleae confirm passwod" v-decorator="[ 'confirmpassword', validatorRules.confirmpassword]"/>
        </a-form-item>

      </a-form>
    </a-spin>
  </a-modal>
</template>

<script>
import md5 from 'md5'
import { changePassword } from '@/api/system'

export default {
  name: 'PasswordModal',
  data () {
    return {
      visible: false,
      confirmLoading: false,
      confirmDirty: false,
      validatorRules: {
        password: {
          rules: [{
            required: true,
            pattern: /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[~!@#$%^&*()_+`\-={}:";'<>?,./]).{8,}$/,
            message: 'The password consists of 8 digits, uppercase and lowercase letters and special symbols.!'
          }, {
            validator: this.validateToNextPassword
          }]
        },
        confirmpassword: {
          rules: [{
            required: true, message: 'Please re-enter your login password!'
          }, {
            validator: this.compareToFirstPassword
          }]
        }
      },

      model: {},

      labelCol: {
        xs: { span: 24 },
        sm: { span: 5 }
      },
      wrapperCol: {
        xs: { span: 24 },
        sm: { span: 16 }
      },
      form: this.$form.createForm(this)
    }
  },
  created () {
    console.log('created')
  },

  methods: {
    show (id, userName) {
      console.log('id', id)
      console.log('userName', userName)
      this.form.resetFields()
      this.visible = true
      this.model.id = id
      this.model.userName = userName
      this.$nextTick(() => {
        this.form.setFieldsValue({ userName: userName })
      })
    },
    close () {
      this.$emit('close')
      this.visible = false
      this.disableSubmit = false
      this.selectedRole = []
    },
    handleSubmit () {
      // 触发表单验证
      this.form.validateFields((err, values) => {
        if (!err) {
          this.confirmLoading = true
          const formData = Object.assign({})
          formData.id = this.model.id
          formData.password = md5(values.password)

          changePassword(formData).then((res) => {
            if (res.success) {
              this.$message.success(res.message)
              this.$emit('ok')
            } else {
              this.$message.warning(res.message)
            }
          }).finally(() => {
            this.confirmLoading = false
            this.close()
          })
        }
      })
    },
    handleCancel () {
      this.close()
    },
    validateToNextPassword  (rule, value, callback) {
      const form = this.form
      const confirmpassword = form.getFieldValue('confirmpassword')
      console.log('confirmpassword==>', confirmpassword)
      if (value && confirmpassword && value !== confirmpassword) {
        callback(new Error('The password entered twice is different!'))
      }
      if (value && this.confirmDirty) {
        form.validateFields(['confirm'], { force: true })
      }
      callback()
    },
    compareToFirstPassword  (rule, value, callback) {
      const form = this.form
      if (value && value !== form.getFieldValue('password')) {
        callback(new Error('The password entered twice is different！'))
      } else {
        callback()
      }
    },
    handleConfirmBlur  (e) {
      const value = e.target.value
      this.confirmDirty = this.confirmDirty || !!value
    }
  }
}
</script>
