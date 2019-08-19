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

        <a-form-item label="Name" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-input placeholder="User's login account name" v-decorator="[ 'username', validatorRules.username]" :readOnly="!!model.id"/>
        </a-form-item>

        <template v-if="!model.id">
          <a-form-item label="Password" :labelCol="labelCol" :wrapperCol="wrapperCol" >
            <a-input type="password" placeholder="User's login account password" v-decorator="[ 'password', validatorRules.password]" />
          </a-form-item>

          <a-form-item label="Confirm password" :labelCol="labelCol" :wrapperCol="wrapperCol" >
            <a-input type="password" @blur="handleConfirmBlur" placeholder="Please confirm password" v-decorator="[ 'confirmpassword', validatorRules.confirmpassword]"/>
          </a-form-item>
        </template>

        <a-form-item label="User Name" :labelCol="labelCol" :wrapperCol="wrapperCol" >
          <a-input placeholder="User's rela name" v-decorator="[ 'realname', validatorRules.realname]" />
        </a-form-item>

        <a-form-item label="Role" :labelCol="labelCol" :wrapperCol="wrapperCol" v-show="!roleDisabled" >
          <a-select
            mode="multiple"
            style="width: 100%"
            placeholder="Please select user's rols"
            v-model="selectedRole">
            <a-select-option v-for="(role,roleindex) in roleList" :key="roleindex.toString()" :value="role.id">
              {{ role.roleName }}
            </a-select-option>
          </a-select>
        </a-form-item>

        <!--部门分配-->
        <a-form-item label="Department" :labelCol="labelCol" :wrapperCol="wrapperCol" v-show="!departDisabled">
          <a-input-search
            placeholder="Click the button on the right to select a department"
            v-model="checkedDepartNameString"
            disabled
            @search="onSearch">
            <a-button slot="enterButton" icon="search">Select</a-button>
          </a-input-search>
        </a-form-item>
        <a-form-item label="Avatar" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-upload
            listType="picture-card"
            class="avatar-uploader"
            :showUploadList="false"
            :action="uploadAction"
            :data="{'isup':1}"
            :headers="headers"
            :beforeUpload="beforeUpload"
            @change="handleChange"
          >
            <img v-if="picUrl" :src="getAvatarView()" alt="Avatar" style="height:104px;max-width:300px"/>
            <div v-else>
              <a-icon :type="uploadLoading ? 'loading' : 'plus'" />
              <div class="ant-upload-text">上传</div>
            </div>
          </a-upload>
        </a-form-item>

        <a-form-item label="Birthday" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-date-picker
            style="width: 100%"
            placeholder="Please select birthday"
            v-decorator="['birthday', {initialValue:!model.birthday?null:moment(model.birthday,dateFormat)}]"/>
        </a-form-item>

        <a-form-item label="Sex" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-select v-decorator="[ 'sex', {}]" placeholder="Please select sex">
            <a-select-option :value="1">male</a-select-option>
            <a-select-option :value="2">female</a-select-option>
          </a-select>
        </a-form-item>

        <a-form-item label="Email" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-input placeholder="Please entry email" v-decorator="[ 'email', validatorRules.email]" />
        </a-form-item>

        <a-form-item label="Phone" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <a-input placeholder="Please entry phone" :disabled="isDisabledAuth('user:form:phone')" v-decorator="[ 'phone', validatorRules.phone]" />
        </a-form-item>

        <a-form-item label="Status" :labelCol="labelCol" :wrapperCol="wrapperCol">
          <ASwitch checkedChildren="Lock" unCheckedChildren="Unlock" defaultChecked="false" onChange="{this.onStatusChange}" />
        </a-form-item>

      </a-form>
    </a-spin>
    <depart-window ref="departWindow" @ok="modalFormOk"></depart-window>

    <div class="drawer-bootom-button" v-show="!disableSubmit">
      <a-popconfirm title="Cancel Edit？" @confirm="handleCancel" okText="Ok" cancelText="Cancel">
        <a-button style="margin-right: .8rem">Cancel</a-button>
      </a-popconfirm>
      <a-button @click="handleSubmit" type="primary" :loading="confirmLoading">Submit</a-button>
    </div>
  </a-drawer>
</template>

<script>
import pick from 'lodash.pick'
import moment from 'moment'
import { addUser, editUser, duplicateCheck } from '@/api/system'
import { getAction } from '@/api/manage'

export default {
  name: 'RoleModal',
  components: {
  },
  data () {
    return {
      departDisabled: false, // 是否是我的部门调用该页面
      roleDisabled: false, // 是否是角色维护调用该页面
      modalWidth: 800,
      drawerWidth: 700,
      modaltoggleFlag: true,
      confirmDirty: false,
      selectedDepartKeys: [], // 保存用户选择部门id
      checkedDepartKeys: [],
      checkedDepartNames: [], // 保存部门的名称 =>title
      checkedDepartNameString: '', // 保存部门的名称 =>title
      userId: '', // 保存用户id
      disableSubmit: false,
      userDepartModel: { userId: '', departIdList: [] }, // 保存SysUserDepart的用户部门中间表数据需要的对象
      dateFormat: 'YYYY-MM-DD',
      validatorRules: {
        username: {
          rules: [{
            required: true, message: 'Please entry account name!'
          }, {
            validator: this.validateUsername
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
        confirmpassword: {
          rules: [{
            required: true, message: 'Please re-enter your login password!'
          }, {
            validator: this.compareToFirstPassword
          }]
        },
        realname: { rules: [{ required: true, message: 'Please enter a username!' }] },
        phone: { rules: [{ validator: this.validatePhone }] },
        email: {
          rules: [{
            validator: this.validateEmail
          }]
        },
        roles: {}
        //  sex:{initialValue:((!this.model.sex)?"": (this.model.sex+""))}
      },
      title: 'Operation',
      visible: false,
      model: {},
      roleList: [],
      selectedRole: [],
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
      picUrl: '',
      url: {
        // fileUpload: window._CONFIG['domianURL'] + '/sys/common/upload',
        // imgerver: window._CONFIG['domianURL'] + '/sys/common/view',
        userWithDepart: '/sys/user/userDepartList', // 引入为指定用户查看部门信息需要的url
        userId: '/sys/user/generateUserId', // 引入生成添加用户情况下的url
        syncUserByUserName: '/process/extActProcess/doSyncUserByUserName'// 同步用户到工作流
      }
    }
  },
  created () {
    // const token = Vue.ls.get(ACCESS_TOKEN)
    // this.headers = { 'X-Access-Token': token }
  },
  computed: {
    uploadAction: function () {
      return this.url.fileUpload
    }
  },
  methods: {
    isDisabledAuth (code) {
      // return disabledAuthFilter(code)
    },
    onStatusChange (checked) {

    },
    // 窗口最大化切换
    toggleScreen () {
      if (this.modaltoggleFlag) {
        this.modalWidth = window.innerWidth
      } else {
        this.modalWidth = 800
      }
      this.modaltoggleFlag = !this.modaltoggleFlag
    },
    initialRoleList () {
      /*
      queryall().then((res) => {
        if (res.success) {
          this.roleList = res.result
        } else {
          console.log(res.message)
        }
      }) */
    },
    loadUserRoles (userid) {
      /*
      queryUserRole({ userid: userid }).then((res) => {
        if (res.success) {
          this.selectedRole = res.result
        } else {
          console.log(res.message)
        }
      }) */
    },
    refresh () {
      this.selectedDepartKeys = []
      this.checkedDepartKeys = []
      this.checkedDepartNames = []
      this.checkedDepartNameString = ''
      this.userId = ''
    },
    add () {
      this.picUrl = ''
      this.refresh()
      this.edit({ activitiSync: '1' })
    },
    edit (record) {
      this.resetScreenSize() // 调用此方法,根据屏幕宽度自适应调整抽屉的宽度
      const that = this
      that.initialRoleList()
      that.checkedDepartNameString = ''
      that.form.resetFields()
      if (record.hasOwnProperty('id')) {
        that.loadUserRoles(record.id)
        this.picUrl = 'Has no pic url yet'
      }
      that.userId = record.id
      that.visible = true
      that.model = Object.assign({}, record)
      that.$nextTick(() => {
        that.form.setFieldsValue(pick(this.model, 'username', 'sex', 'realname', 'email', 'phone', 'activitiSync'))
      })
      // 调用查询用户对应的部门信息的方法
      that.checkedDepartKeys = []
      that.loadCheckedDeparts()
    },

    loadCheckedDeparts () {
      const that = this
      if (!that.userId) { return }
      getAction(that.url.userWithDepart, { userId: that.userId }).then((res) => {
        that.checkedDepartNames = []
        if (res.success) {
          for (let i = 0; i < res.result.length; i++) {
            that.checkedDepartNames.push(res.result[i].title)
            this.checkedDepartNameString = this.checkedDepartNames.join(',')
            that.checkedDepartKeys.push(res.result[i].key)
          }
          that.userDepartModel.departIdList = that.checkedDepartKeys
        } else {
          console.log(res.message)
        }
      })
    },
    close () {
      this.$emit('close')
      this.visible = false
      this.disableSubmit = false
      this.selectedRole = []
      this.userDepartModel = { userId: '', departIdList: [] }
      this.checkedDepartNames = []
      this.checkedDepartNameString = ''
      this.checkedDepartKeys = []
      this.selectedDepartKeys = []
    },
    moment,
    handleSubmit () {
      const that = this
      // 触发表单验证
      this.form.validateFields((err, values) => {
        if (!err) {
          that.confirmLoading = true
          const avatar = that.model.avatar
          if (!values.birthday) {
            values.birthday = ''
          } else {
            values.birthday = values.birthday.format(this.dateFormat)
          }
          const formData = Object.assign(this.model, values)
          formData.avatar = avatar
          formData.selectedroles = this.selectedRole.length > 0 ? this.selectedRole.join(',') : ''
          formData.selecteddeparts = this.userDepartModel.departIdList.length > 0 ? this.userDepartModel.departIdList.join(',') : ''

          // that.addDepartsToUser(that,formData); // 调用根据当前用户添加部门信息的方法
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
            that.checkedDepartNames = []
            that.userDepartModel.departIdList = { userId: '', departIdList: [] }

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
      const confirmpassword = form.getFieldValue('confirmpassword')

      if (value && confirmpassword && value !== confirmpassword) {
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
    validateUsername (rule, value, callback) {
      var params = {
        tableName: 'sys_user',
        fieldName: 'username',
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
    beforeUpload: function (file) {
      var fileType = file.type
      if (fileType.indexOf('image') < 0) {
        this.$message.warning('Please upload image')
        return false
      }
      // TODO Verify image file size
    },
    handleChange (info) {
      this.picUrl = ''
      if (info.file.status === 'uploading') {
        this.uploadLoading = true
        return
      }
      if (info.file.status === 'done') {
        var response = info.file.response
        this.uploadLoading = false
        console.log(response)
        if (response.success) {
          this.model.avatar = response.message
          this.picUrl = 'Has no pic url yet'
        } else {
          this.$message.warning(response.message)
        }
      }
    },
    getAvatarView () {
      return this.url.imgerver + '/' + this.model.avatar
    },
    // 搜索用户对应的部门API
    onSearch () {
      this.$refs.departWindow.add(this.checkedDepartKeys, this.userId)
    },

    // 获取用户对应部门弹出框提交给返回的数据
    modalFormOk (formData) {
      this.checkedDepartNames = []
      this.selectedDepartKeys = []
      this.checkedDepartNameString = ''
      this.userId = formData.userId
      this.userDepartModel.userId = formData.userId
      for (let i = 0; i < formData.departIdList.length; i++) {
        this.selectedDepartKeys.push(formData.departIdList[i].key)
        this.checkedDepartNames.push(formData.departIdList[i].title)
        this.checkedDepartNameString = this.checkedDepartNames.join(',')
      }
      this.userDepartModel.departIdList = this.selectedDepartKeys
      this.checkedDepartKeys = this.selectedDepartKeys // 更新当前的选择keys
    },
    // 根据屏幕变化,设置抽屉尺寸
    resetScreenSize () {
      const screenWidth = document.body.clientWidth
      if (screenWidth < 500) {
        this.drawerWidth = screenWidth
      } else {
        this.drawerWidth = 700
      }
    }
  }
}
</script>

<style scoped>
  .avatar-uploader > .ant-upload {
    width:104px;
    height:104px;
  }
  .ant-upload-select-picture-card i {
    font-size: 49px;
    color: #999;
  }

  .ant-upload-select-picture-card .ant-upload-text {
    margin-top: 8px;
    color: #666;
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
