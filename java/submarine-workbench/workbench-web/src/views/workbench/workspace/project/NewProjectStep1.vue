<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<template>
  <div style="padding-top: 10px;">
    <a-form :form="form" style="max-width: 650px; margin: 10px auto 0;">

      <a-form-item label="Project Name" :labelCol="labelCol" :wrapperCol="wrapperCol">
        <a-input v-model="project.name" v-decorator="[ 'name', validatorRules.name]" />
      </a-form-item>

      <a-form-item label="Description" :labelCol="labelCol" :wrapperCol="wrapperCol">
        <a-textarea :rows="4" v-model="project.description" v-decorator="['desc', validatorRules.description]" />
      </a-form-item>

      <a-form-item label="Visibility" :labelCol="labelCol" :wrapperCol="wrapperCol">
        <a-radio-group v-model="project.visibility" @change="visibilityOnChange" style="width: 100%">
          <a-radio class="radioStyle" :value="'PROJECT_VISIBILITY_PRIVATE'"><a class="a-radio_a">Private</a> - Only project collaborators can view or edit the project. </a-radio>
          <a-radio class="radioStyle" :value="'PROJECT_VISIBILITY_TEAM'"><a class="a-radio_a">Team</a> - All members of the team can view the project.</a-radio>
          <dict-select-tag
            v-if="project.visibility==='PROJECT_VISIBILITY_TEAM'"
            tableName="team"
            triggerChange="true"
            @change="teamNameOnChange"
            v-model="project.teamName"
            v-decorator="['teamName', validatorRules.teamName]"/>
          <a-radio class="radioStyle" :value="'PROJECT_VISIBILITY_PUBLIC'"><a class="a-radio_a">Public</a> - All authenticated users can view the project.</a-radio>
        </a-radio-group>
      </a-form-item>

      <a-form-item label="Permission" :labelCol="labelCol" :wrapperCol="wrapperCol" v-if="project.visibility!=='PROJECT_VISIBILITY_PRIVATE'">
        <a-radio-group v-model="project.permission" @change="permissionOnChange" style="width: 100%">
          <a-radio class="radioStyle" :value="'PROJECT_PERMISSION_VIEW'"><a class="a-radio_a">Can View</a>{{ permissionCanView }}</a-radio>
          <a-radio class="radioStyle" :value="'PROJECT_PERMISSION_EDIT'"><a class="a-radio_a">Can Edit</a>{{ permissionCanEdit }}</a-radio>
          <a-radio class="radioStyle" :value="'PROJECT_PERMISSION_EXECUTE'"><a class="a-radio_a">Can Execute</a>{{ permissionCanExecute }}</a-radio>
        </a-radio-group>
      </a-form-item>

      <a-form-item style="text-align: center; margin-top: 40px;">
        <a-button type="primary" @click="nextStep">Next Step<a-icon type="right" /></a-button>
      </a-form-item>

    </a-form>
  </div>
</template>

<script>
import pick from 'lodash.pick'
import DictSelectTag from '@/components/Dict/DictSelectTag.vue'

export default {
  name: 'NewProjectStep1',
  components: { DictSelectTag },

  props: {
    project: {
      type: Object,
      // Object or array defaults must be obtained from a factory function
      default: function () {
        return { }
      },
      required: true
    }
  },

  model: {
    // Pass the variable value to the child component when the parent component sets the v-model
    prop: 'project'
  },
  mounted () {
    console.log('projectaaa', this.project)
    const that = this
    that.form.resetFields()
    that.$nextTick(() => {
      that.form.setFieldsValue(pick(this.project, 'name', 'description', 'visibility', 'teamName', 'permission'))
    })
    that.setPermissionDesc()
  },

  data () {
    return {
      labelCol: { lg: { span: 5 }, sm: { span: 5 } },
      wrapperCol: { lg: { span: 19 }, sm: { span: 19 } },
      validatorRules: {
        name: { rules: [{ required: true, message: 'Please enter project name!' }] },
        description: { rules: [{ required: true, message: 'Please enter project description!' }] },
        teamName: { rules: [{ required: true, message: 'Please select team name!' }] }
      },
      permissionCanView: '',
      permissionCanEdit: '',
      permissionCanExecute: '',
      form: this.$form.createForm(this)
    }
  },

  methods: {
    nextStep () {
      const { form: { validateFields } } = this
      // Check the form, then go to the next step.
      validateFields((err, values) => {
        if (!err) {
          console.log('project=', this.project)
          this.$emit('nextStep', this.project)
        }
      })
    },
    visibilityOnChange (e) {
      this.project.visibility = e.target.value
      console.log('project.visibility=', this.project.visibility)
      if (this.project.visibility !== 'PROJECT_VISIBILITY_TEAM') {
        this.project.teamName = ''
      }
      this.setPermissionDesc()
      this.$emit('updateProject', this.project)
    },
    setPermissionDesc () {
      if (this.project.visibility !== 1) {
        this.permissionCanView = ' - All members can view the project.'
        this.permissionCanEdit = ' - All members can edit the project.'
        this.permissionCanExecute = ' - All members can execute the project.'
      } else {
        this.permissionCanView = ' - All members of the team can view the project.'
        this.permissionCanEdit = ' - All members of the team can edit the project.'
        this.permissionCanExecute = ' - All members of the team can execute the project.'
      }
    },
    permissionOnChange (e) {
      this.project.permission = e.target.value
      this.$emit('updateProject', this.project)
    },
    teamNameOnChange (val) {
      this.project.teamName = val
      this.$emit('updateProject', this.project)
    }
  }
}
</script>

<style lang="less" scoped>
  .a-radio_a {
    font-size: 14px;
    font-family: bold;
    color: rgba(0, 0, 0, 1);
  }

  .radioStyle {
    display: block;
    height: 30px;
    lineHeight: 30px;
  }

  .tab-content {
    margin-top: 4px;
    border: 1px dashed #e9e9e9;
    border-radius: 6px;
    background-color: #fafafa;
    min-height: 200px;
    // text-align: center;
    // padding-top: 30px;

    p {
      padding-top: 80px;
      font-size: 20px;
    }
  }

  .step-form-style-desc {
    padding: 0 56px;
    color: rgba(0,0,0,.45);

    h3 {
      margin: 0 0 12px;
      color: rgba(0,0,0,.45);
      font-size: 16px;
      line-height: 32px;
    }

    h4 {
      margin: 0 0 4px;
      color: rgba(0,0,0,.45);
      font-size: 14px;
      line-height: 22px;
    }

    p {
      margin-top: 0;
      margin-bottom: 12px;
      line-height: 22px;
    }
  }
</style>
