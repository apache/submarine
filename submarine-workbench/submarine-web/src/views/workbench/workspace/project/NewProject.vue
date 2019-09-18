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
    <a-button shape="circle" icon="close" slot="extra" @click="showProjectList"/>

    <a-spin :spinning="confirmLoading">
      <a-steps :current="currentStep" class="steps">
        <a-step title="Basic Information" />
        <a-step title="Initial Setup" />
        <a-step title="Preview Project" />
      </a-steps>
      <a-divider type="horizontal" style="width: 100%"/>
      <div class="content">
        <step1 v-if="currentStep === 0" @nextStep="nextStep" />
        <step2 v-if="currentStep === 1" @nextStep="nextStep" @prevStep="prevStep"/>
        <step3 v-if="currentStep === 2" @prevStep="prevStep" @finish="finish"/>
      </div>
    </a-spin>
  </a-card>
</template>

<script>
import pick from 'lodash.pick'
import Step1 from './NewProjectStep1'
import Step2 from './NewProjectStep2'
import Step3 from './NewProjectStep3'

const stepForms = [
  ['name', 'desc'],
  ['target', 'template', 'type'],
  ['time', 'frequency']
]

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
      mdl: {},
      radioStyle: {
        display: 'block',
        height: '30px',
        lineHeight: '30px'
      },
      form: this.$form.createForm(this)
    }
  },
  methods: {
    showProjectList () {
      this.$emit('showProjectList')
    },
    nextStep () {
      if (this.currentStep < 2) {
        this.currentStep += 1
      }
    },
    prevStep () {
      if (this.currentStep > 0) {
        this.currentStep -= 1
      }
    },
    finish () {
      this.$emit('showProjectList')
      this.currentStep = 0
    },
    edit (record) {
      this.visible = true
      const { form: { setFieldsValue } } = this
      this.$nextTick(() => {
        setFieldsValue(pick(record, []))
      })
    },
    handleNext (step) {
      const { form: { validateFields } } = this
      const currentStep = step + 1
      if (currentStep <= 2) {
        // stepForms
        validateFields(stepForms[ this.currentStep ], (errors, values) => {
          if (!errors) {
            this.currentStep = currentStep
          }
        })
        return
      }
      // last step
      this.confirmLoading = true
      validateFields((errors, values) => {
        console.log('errors:', errors, 'val:', values)
        if (!errors) {
          console.log('values:', values)
          setTimeout(() => {
            this.confirmLoading = false
            this.$emit('ok', values)
          }, 1500)
        } else {
          this.confirmLoading = false
        }
      })
    },
    backward () {
      this.currentStep--
    },
    handleCancel () {
      // clear form & currentStep
      this.visible = false
      this.currentStep = 0
    },
    handleChange (info) {
      const status = info.file.status
      if (status !== 'uploading') {
        console.log(info.file, info.fileList)
      }
      if (status === 'done') {
        this.$message.success(`${info.file.name} file uploaded successfully.`)
      } else if (status === 'error') {
        this.$message.error(`${info.file.name} file upload failed.`)
      }
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
