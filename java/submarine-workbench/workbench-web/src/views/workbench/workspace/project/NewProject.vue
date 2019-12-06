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
  <a-card title="Create New Project">
    <a-button type="primary" shape="circle" icon="close" slot="extra" @click="showProjectList"/>

    <a-spin :spinning="confirmLoading">
      <a-steps :current="currentStep" class="steps">
        <a-step title="Basic Information" />
        <a-step title="Initial Project" />
        <a-step title="Preview Project" />
      </a-steps>
      <a-divider type="horizontal" style="width: 100%"/>
      <div class="content">
        <step1 v-if="currentStep === 0" v-model="project" @nextStep="nextStep" @updateProject="updateProject"/>
        <step2 v-if="currentStep === 1" v-model="project" @nextStep="nextStep" @prevStep="prevStep" @updateProject="updateProject"/>
        <step3 v-if="currentStep === 2" v-model="project" @prevStep="prevStep" @finish="finish" @updateProject="updateProject"/>
      </div>
    </a-spin>
  </a-card>
</template>

<script>
import pick from 'lodash.pick'
import Step1 from './NewProjectStep1'
import Step2 from './NewProjectStep2'
import Step3 from './NewProjectStep3'
import { addProject } from '@/api/system'

export default {
  name: 'StepByStepModal',
  components: {
    Step1,
    Step2,
    Step3
  },
  data () {
    return {
      labelCol: {
        xs: { span: 24 },
        sm: { span: 7 }
      },
      wrapperCol: {
        xs: { span: 24 },
        sm: { span: 13 }
      },
      visible: false,
      confirmLoading: false,
      currentStep: 0,
      project: {
        name: '',
        userName: '',
        description: '',
        type: 'PROJECT_TYPE_NOTEBOOK',
        teamName: '',
        visibility: 'PROJECT_VISIBILITY_PRIVATE',
        permission: 'PROJECT_PERMISSION_VIEW',
        starNum: 0,
        likeNum: 0,
        messageNum: 0
      },
      radioStyle: {
        display: 'block',
        height: '30px',
        lineHeight: '30px'
      },
      login_user: {},
      form: this.$form.createForm(this)
    }
  },
  computed: {
    userInfo () {
      return this.$store.getters.userInfo
    }
  },
  created () {
    this.login_user = this.userInfo
  },
  methods: {
    showProjectList () {
      this.$emit('showProjectList')
    },
    updateProject: function (childProject) {
      console.log('updateProject=', childProject)
      this.project = Object.assign(this.project, childProject)
    },
    nextStep: function (childModel) {
      console.log('childModel=', childModel)
      this.project = Object.assign(this.project, childModel)
      console.log('project = ', this.project)
      if (this.currentStep < 2) {
        this.currentStep += 1
      }
    },
    prevStep: function (childModel) {
      console.log('childModel=', childModel)
      this.project = Object.assign(this.project, childModel)
      console.log('project = ', this.project)
      if (this.currentStep > 0) {
        this.currentStep -= 1
      }
    },
    finish: function (childModel) {
      this.project = Object.assign(this.project, childModel)
      this.project.userName = this.login_user.name
      this.project.createBy = this.login_user.name
      this.project.updateBy = this.login_user.name

      addProject(this.project).then((res) => {
        if (res.success) {
          this.$message.info(res.message)
          this.$emit('showProjectList')
          this.currentStep = 0
        } else {
          this.$message.warning(res.message)
        }
      })
    },
    edit (record) {
      this.visible = true
      const { form: { setFieldsValue } } = this
      this.$nextTick(() => {
        setFieldsValue(pick(record, []))
      })
    }
  }
}
</script>
<style lang="less" scoped>

  .steps {
    max-width: 700px;
    margin: 20px auto 30px;
  }

  .table-operations {
    margin-bottom: 16px;
  }

  .table-operations > button {
    margin-right: 8px;
  }
</style>
