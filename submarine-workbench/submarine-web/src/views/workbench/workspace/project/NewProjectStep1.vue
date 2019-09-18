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
        <a-input v-decorator="['name', {rules: [{required: true}]}]" />
      </a-form-item>

      <a-form-item label="Description" :labelCol="labelCol" :wrapperCol="wrapperCol">
        <a-textarea :rows="4" v-decorator="['desc', {rules: [{required: true}]}]"></a-textarea>
      </a-form-item>

      <a-form-item label="Visibility" :labelCol="labelCol" :wrapperCol="wrapperCol">
        <a-radio-group v-decorator="['type', {initialValue: 0, rules: [{required: true}]}]" style="width: 100%">
          <a-radio class="radioStyle" :value="0"><a class="a-radio_a">Private</a> - Only project collaborators can view or edit the project. </a-radio>
          <a-radio class="radioStyle" :value="1"><a class="a-radio_a">Team</a> - All members of the team can view the project.</a-radio>
          <a-radio class="radioStyle" :value="2"><a class="a-radio_a">Public</a> - All authenticated users can view the project.</a-radio>
        </a-radio-group>
      </a-form-item>

      <a-form-item style="text-align: center; margin-top: 40px;">
        <a-button type="primary" @click="nextStep">Next Step<a-icon type="right" /></a-button>
      </a-form-item>

    </a-form>
  </div>
</template>

<script>

export default {
  name: 'Step1',
  data () {
    return {
      labelCol: { lg: { span: 5 }, sm: { span: 5 } },
      wrapperCol: { lg: { span: 19 }, sm: { span: 19 } },
      form: this.$form.createForm(this),
      fileList: [],
      uploading: false,
      dataSourceType: 'file'
    }
  },
  methods: {
    nextStep () {
      const { form: { validateFields } } = this
      // 先校验，通过表单校验后，才进入下一步
      validateFields((err, values) => {
        if (!err) {
          this.$emit('nextStep')
        }
      })
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
