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
  <div style="margin: 20px auto 0;">
    <a-tabs defaultActiveKey="1" tab-position="top" type="card">
      <a-tab-pane key="1">
        <span slot="tab">
          <a-icon type="file" />
          Blank
        </span>
        <div class="tab-content"><p>You can either create a blank project.</p></div>
      </a-tab-pane>

      <a-tab-pane key="2">
        <span slot="tab">
          <a-icon type="database" />
          Template
        </span>
        <a-table
          :columns="templateColumns"
          :dataSource="templateDataSource"
          :pagination="false"
          :scroll="{ y: 300 }"
          bordered
        >
        </a-table>
      </a-tab-pane>
      <a-tab-pane key="3">
        <span slot="tab">
          <a-icon type="upload" />
          Upload
        </span>
        <a-upload-dragger name="file" :multiple="true" action="https://www.mocky.io/v2/5cc8019d300000980a055e76">
          <p class="ant-upload-drag-icon">
            <a-icon type="inbox" />
          </p>
          <p class="ant-upload-text">Click or drag file to this area to upload</p>
          <p class="ant-upload-hint">Support for a single or bulk upload. Strictly prohibit from uploading company data or other band files</p>
        </a-upload-dragger>
      </a-tab-pane>
      <a-tab-pane key="4">
        <span slot="tab">
          <a-icon type="github" />
          Git Repo
        </span>
        <a-table
          :columns="gitColumns"
          :dataSource="gitDataSource"
          :pagination="false"
          :scroll="{ y: 300 }"
          bordered
        >
        </a-table>
      </a-tab-pane>
    </a-tabs>

    <div style="text-align: center; margin-top: 40px;">
      <a-button @click="prevStep"><a-icon type="left" />Previous Step</a-button>
      <a-button type="primary" @click="nextStep" style="margin-left: 10px;">Next Step<a-icon type="right" /></a-button>
    </div>
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
      templateColumns: [
        {
          title: 'Type',
          dataIndex: 'type',
          key: 'type',
          width: 250
        },
        {
          title: 'Description',
          dataIndex: 'description',
          key: 'description'
        }
      ],
      templateDataSource: [
        {
          key: 'op1',
          type: 'Python',
          description: 'Python template'
        },
        {
          key: 'op2',
          type: 'R',
          description: 'Python template'
        },
        {
          key: 'op3',
          type: 'Spark',
          description: 'Spark template'
        },
        {
          key: 'op4',
          type: 'Tensorflow',
          description: 'Tensorflow template'
        },
        {
          key: 'op5',
          type: 'PyTorch',
          description: 'PyTorch template'
        }
      ],

      gitColumns: [
        {
          title: 'Name',
          dataIndex: 'name',
          key: 'name',
          width: 250
        },
        {
          title: 'Description',
          dataIndex: 'description',
          key: 'description'
        }
      ],
      gitDataSource: [
        {
          key: 'op1',
          name: 'Python test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op2',
          name: 'R test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op3',
          name: 'Spark test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op4',
          name: 'Tensorflow test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op5',
          name: 'PyTorch test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op6',
          name: 'PyTorch test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op7',
          name: 'PyTorch test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op8',
          name: 'PyTorch test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op9',
          name: 'PyTorch test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        },
        {
          key: 'op10',
          name: 'PyTorch test',
          description: 'Apache {Submarine} is the latest machine learning framework'
        }
      ]
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
    },
    prevStep () {
      this.$emit('prevStep')
    }
  }
}
</script>

<style lang="less" scoped>
  .tab-content {
    margin-top: 4px;
    border: 1px dashed #e9e9e9;
    border-radius: 6px;
    background-color: #fafafa;
    min-height: 200px;
    text-align: center;
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
