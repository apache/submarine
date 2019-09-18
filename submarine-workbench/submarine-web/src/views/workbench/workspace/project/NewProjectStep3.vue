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
  <div>
    <a-row style="margin-bottom: 8px; margin-left: 8px; margin-right: 8px;">
      <a-col :span="6"><span style="font-size: 18px; font-weight: bold">Files</span></a-col>
      <a-col :span="12"></a-col>
      <a-col :span="6" style="text-align: right">
      </a-col>
    </a-row>

    <a-table
      :columns="projectColumns"
      :dataSource="projectDataSource"
      :pagination="false"
      :scroll="{ y: 300 }"
      bordered
    >

      <template slot="footer">
        Total: 35 files
      </template>
    </a-table>

    <div style="text-align: center; margin-top: 40px;">
      <a-button @click="prevStep"><a-icon type="left" />Previous Step</a-button>
      <a-button style="margin-left: 8px" type="primary" @click="finish" icon="check">Submit</a-button>
      <a-button style="margin-left: 8px" type="primary" icon="form">Open Workbench</a-button>
    </div>
  </div>
</template>

<script>
import { STable } from '@/components'

export default {
  name: 'DataPage',
  components: {
    STable
  },
  data () {
    return {
      labelCol: { lg: { span: 5 }, sm: { span: 5 } },
      wrapperCol: { lg: { span: 19 }, sm: { span: 19 } },
      form: this.$form.createForm(this),
      projectColumns: [
        {
          title: 'Name',
          dataIndex: 'name',
          sorter: true,
          key: 'name'
        },
        {
          title: 'Size',
          dataIndex: 'size',
          width: 100,
          key: 'size'
        },
        {
          title: 'Last Modified',
          dataIndex: 'lastModified',
          width: 200,
          key: 'lastModified'
        }
      ],
      projectDataSource: [
        {
          key: 'op1',
          name: 'Test data',
          size: '1.2 MB',
          lastModified: '2019-09-01 12:00:00',
          children: [{
            key: 'key-01-01',
            title: 'Python',
            icon: 'folder-open'
          }, {
            key: 'key-01-02',
            name: 'Test data',
            size: '1.2 MB',
            icon: 'folder-open'
          }, {
            key: 'key-01-03',
            name: 'Test data',
            size: '1.2 MB',
            icon: 'folder-open',
            children: [{
              key: 'key-01-03-01',
              name: 'Test data',
              size: '1.2 MB',
              icon: 'folder-open'
            },
            {
              key: 'key-01-03-02',
              name: 'Test data',
              size: '1.2 MB',
              icon: 'folder-open'
            }
            ]
          }]
        },
        {
          key: 'op2',
          name: 'Test data',
          size: '1.2 MB',
          lastModified: '2019-09-01 12:00:00'
        },
        {
          key: 'op3',
          name: 'Test data',
          size: '1.2 MB',
          lastModified: '2019-09-01 12:00:00'
        },
        {
          key: 'op4',
          name: 'Test data',
          size: '1.2 MB',
          lastModified: '2019-09-01 12:00:00'
        },
        {
          key: 'op5',
          name: 'Test data',
          size: '1.2 MB',
          lastModified: '2019-09-01 12:00:00'
        }
      ]
    }
  },
  created () {

  },
  methods: {
    prevStep () {
      this.$emit('prevStep')
    },
    finish () {
      this.$emit('finish')
    }
  }
}
</script>

<style lang="less" scoped>
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
