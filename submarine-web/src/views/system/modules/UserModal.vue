<template>
  <a-drawer
    :title="title"
    :maskClosable="true"
    :width="drawerWidth"
    placement="right"
    :closable="true"
    @close="handleCancel"
    :visible="visible"
    cancelText="Close"
    okText="Ok"
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

        <a-form-item label="Account Name" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-input placeholder="User's account name" v-decorator="[ 'userName', validatorRules.userName]" :readOnly="!!model.id"/>
        </a-form-item>

        <template v-if="!model.id">
          <a-form-item label="Password" :labelCol="labelCol" :wrapperCol="wrapperCol" >
            <a-input type="password" placeholder="User's account password" v-decorator="[ 'password', validatorRules.password]" />
          </a-form-item>

          <a-form-item label="Confirm" :labelCol="labelCol" :wrapperCol="wrapperCol" >
            <a-input type="password" @blur="handleConfirmBlur" placeholder="Please confirm password" v-decorator="[ 'confirmPassword', validatorRules.confirmPassword]"/>
          </a-form-item>
        </template>

        <a-form-item label="Real Name" :labelCol="labelCol" :wrapperCol="wrapperCol" >
          <a-input placeholder="User's real name" v-decorator="[ 'realName', validatorRules.realName]" />
        </a-form-item>

        <a-form-item label="Department" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-tree-select
            style="width:100%"
            :dropdownStyle="{maxHeight:'200px',overflow:'auto'}"
            :treeData="treeSelectData"
            showSearch
            allowClear
            treeDefaultExpandAll
            v-model="model.deptCode"
            @change="onChangeDeptCode">
          </a-tree-select>
        </a-form-item>

        <a-form-item label="Birthday" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-date-picker
            style="width: 100%"
            placeholder="Please select birthday"
            v-decorator="['birthday', {initialValue:!model.birthday?null:moment(model.birthday,dateFormat)}]"/>
        </a-form-item>

        <a-form-item label="Sex" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <dict-select-tag dictCode="SYS_USER_SEX" v-model="model.sex"/>
        </a-form-item>

        <a-form-item label="Email" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-input placeholder="Please entry email" v-decorator="[ 'email', validatorRules.email]" />
        </a-form-item>

        <a-form-item label="Phone" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-input placeholder="Please entry phone" v-decorator="[ 'phone', validatorRules.phone]" />
        </a-form-item>

        <a-form-item label="Status" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <dict-select-tag dictCode="SYS_USER_STATUS" v-model="model.status"/>
        </a-form-item>

      </a-form>
    </a-spin>

    <div class="drawer-bootom-button" v-show="!disableSubmit">
      <a-button style="margin-right: .8rem" @click="handleCancel">Cancel</a-button>
      <a-button @click="handleSubmit" type="primary" :loading="confirmLoading">Submit</a-button>
    </div>
  </a-drawer>
</template>

<script>
import md5 from 'md5'
import pick from 'lodash.pick'
import moment from 'moment'
import { addUser, editUser, duplicateCheck, queryIdTree } from '@/api/system'
import DictSelectTag from '@/components/Dict/DictSelectTag.vue'

export default {
  name: 'UserModal',
  components: {
    DictSelectTag
  },
  data () {
    return {
      treeSelectData: [],
      modalWidth: 800,
      drawerWidth: 700,
      modaltoggleFlag: true,
      confirmDirty: false,
      userId: '', // 保存用户id
      disableSubmit: false,
      dateFormat: 'YYYY-MM-DD HH:mm:ss',
      validatorRules: {
        userName: {
          rules: [{
            required: true, message: 'Please entry user name!'
          }, {
            validator: this.validateUserName
          }]
        },
        password: {
          rules: [{
            required: true,
            pattern: /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[~!@#$%^&*()_+`\-={}:";'<>?,./]).{8,}$/,
            message: 'The password consists of 8 digits, uppercase and lowercase letters and special symbols.'
          }, {
            validator: this.validateToNextPassword
          }]
        },
        confirmPassword: {
          rules: [{
            required: true, message: 'Please re-enter your login password!'
          }, {
            validator: this.compareToFirstPassword
          }]
        },
        realName: { rules: [{ required: true, message: 'Please enter real name!' }] },
        phone: { rules: [{ validator: this.validatePhone }] },
        email: {
          rules: [{
            validator: this.validateEmail
          }]
        }
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
      uploadLoading: false,
      confirmLoading: false,
      headers: {},
      form: this.$form.createForm(this),
      url: {
      }
    }
  },
  created () {
    // const token = Vue.ls.get(ACCESS_TOKEN)
    // this.headers = { 'X-Access-Token': token }
    this.loadTreeSelectData()
  },
  computed: {

  },
  methods: {
    onStatusChange (checked) {

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
    refresh () {
      this.userId = ''
    },
    add () {
      this.refresh()
      this.edit({ })
    },
    edit (record) {
      // Call this method to adaptively adjust the width of the drawer according to the width of the screen
      this.resetScreenSize()
      const that = this
      that.form.resetFields()
      that.userId = record.id
      that.visible = true
      that.model = Object.assign({}, record)
      that.$nextTick(() => {
        that.form.setFieldsValue(pick(this.model, 'userName', 'sex', 'realName', 'email', 'phone', 'password'))
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
      // Trigger form validation
      this.form.validateFields((err, values) => {
        if (!err) {
          that.confirmLoading = true
          if (!values.birthday) {
            values.birthday = null
          } else {
            values.birthday = moment(values.birthday).format(this.dateFormat)
          }
          const formData = Object.assign(this.model, values)
          if (values.password) {
            formData.password = md5(values.password)
          }
          // because SysUser not contain @dict field, So need delete from object
          if (formData.hasOwnProperty('sex@dict')) {
            // need delete dict Annotation
            delete formData['sex@dict']
          }
          if (formData.hasOwnProperty('status@dict')) {
            // need delete dict Annotation
            delete formData['status@dict']
          }
          if (formData.hasOwnProperty('confirmPassword')) {
            // need delete dict Annotation
            delete formData['confirmPassword']
          }

          let obj
          if (!this.model.id) {
            formData.id = this.userId
            obj = addUser(formData)
          } else {
            obj = editUser(formData)
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
    handleCancel () {
      this.close()
    },
    validateToNextPassword  (rule, value, callback) {
      const form = this.form
      const confirmPassword = form.getFieldValue('confirmPassword')

      if (value && confirmPassword && value !== confirmPassword) {
        callback(new Error('The password entered twice is different！'))
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
    validatePhone (rule, value, callback) {
      if (!value) {
        callback()
      } else {
        if (new RegExp(/^1[3|4|5|7|8][0-9]\d{8}$/).test(value)) {
          var params = {
            tableName: 'sys_user',
            fieldName: 'phone',
            fieldVal: value,
            dataId: this.userId
          }
          duplicateCheck(params).then((res) => {
            if (res.success) {
              callback()
            } else {
              callback(new Error('The phone number already exists!'))
            }
          })
        } else {
          callback(new Error('Please enter the phone number in the correct format!'))
        }
      }
    },
    validateEmail (rule, value, callback) {
      if (!value) {
        callback()
      } else {
        if (new RegExp(/^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$/).test(value)) {
          var params = {
            tableName: 'sys_user',
            fieldName: 'email',
            fieldVal: value,
            dataId: this.userId
          }
          duplicateCheck(params).then((res) => {
            console.log(res)
            if (res.success) {
              callback()
            } else {
              callback(new Error('The email already exists!'))
            }
          })
        } else {
          callback(new Error('Please enter the email in the correct format!'))
        }
      }
    },
    validateUserName (rule, value, callback) {
      var params = {
        tableName: 'sys_user',
        fieldName: 'user_name',
        fieldVal: value,
        dataId: this.userId
      }
      duplicateCheck(params).then((res) => {
        if (res.success) {
          callback()
        } else {
          callback(new Error('Account name already exist!'))
        }
      })
    },
    handleConfirmBlur  (e) {
      const value = e.target.value
      this.confirmDirty = this.confirmDirty || !!value
    },

    normFile  (e) {
      console.log('Upload event:', e)
      if (Array.isArray(e)) {
        return e
      }
      return e && e.fileList
    },
    // 搜索用户对应的部门API
    onSearch () {
    },
    // 根据屏幕变化,设置抽屉尺寸
    resetScreenSize () {
      const screenWidth = document.body.clientWidth
      if (screenWidth < 500) {
        this.drawerWidth = screenWidth
      } else {
        this.drawerWidth = 700
      }
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
    }
  }
}
</script>

<style scoped>
  .ant-upload-select-picture-card i {
    font-size: 49px;
    color: #999;
  }

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
