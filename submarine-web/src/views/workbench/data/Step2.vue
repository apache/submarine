<template>
  <div>
    <a-form :form="form" style="max-width: 650px; margin: 40px auto 0;">
      <a-form-item
        label="Database Name"
        :labelCol="labelCol"
        :wrapperCol="wrapperCol"
      >
        <a-select placeholder="Please select database">
          <a-select-option value="db1">db1</a-select-option>
          <a-select-option value="db2">db2</a-select-option>
          <a-select-option value="db3">db3</a-select-option>
        </a-select>
      </a-form-item>

      <a-form-item
        label="Table Name"
        :labelCol="labelCol"
        :wrapperCol="wrapperCol"
      >
        <a-input v-decorator="['name', { initialValue: '', rules: [{required: true, message: 'Must enter the table name!'}] }]"/>
      </a-form-item>

      <a-form-item
        :labelCol="labelCol"
        :wrapperCol="wrapperCol"
        label="Parse Columns">
        <a-button @click="parseColumns" icon="sync">Parse</a-button>
      </a-form-item>
      <a-form-item>
        <a-table
          ref="table1"
          size="default"
          :columns="schemaColumns"
          :dataSource="schemaDataSource"
          :pagination="false"
          bordered
        >
          <template slot="title">
            <a-button @click="handleAddColumn" icon="plus">Add New Column</a-button>
          </template>
          <template v-for="col in ['name', 'type', 'comment']" :slot="col" slot-scope="text, record">
            <div :key="col">
              <a-input
                v-if="record.editable"
                style="margin: -5px 0"
                :value="text"
                @change="e => onChangeColumn(e.target.value, record.key, col)"
              />
              <template v-else>{{ text }}</template>
            </div>
          </template>
          <template slot="operation" slot-scope="text, record">
            <span>
              <span v-if="record.editable">
                <a @click="() => onSaveColumn(record.key)">Save</a> or
                <a-popconfirm title="Sure to cancel?" @confirm="() => onCancelColumn(record.key)">
                  <a>Cancel</a>
                </a-popconfirm>
              </span>
              <span v-else>
                <a @click="() => onEditColumn(record.key)">Edit</a>
              </span>
              <a-divider type="vertical" />
              <a-popconfirm
                v-if="schemaDataSource.length"
                title="Sure to delete?"
                @confirm="() => handleDelColumn(record.key)">
                <a href="javascript:;">Delete</a>
              </a-popconfirm>
            </span>
          </template>
        </a-table>
      </a-form-item>

      <a-form-item :wrapperCol="{span: 19, offset: 5}">
        <a-button @click="prevStep"><a-icon type="left" />Previous Step</a-button>
        <a-button style="margin-left: 8px" type="primary" @click="nextStep"><a-icon type="right" />Next Step</a-button>
      </a-form-item>
    </a-form>
    <a-divider />
    <div class="step-form-style-desc">
      <h3>Description</h3>
      <h4>Database Name</h4>
      <p>Select which database belongs to which table you want to create.</p>
      <h4>Table Name</h4>
      <p>Set the name of the table and automatically check if the table name conflicts.</p>
      <h4>Parse Columns</h4>
      <p>Click the Parse button to analyze the schema field and type of the table from the uploaded file.</p>
      <h4>Columns Attributes Table</h4>
      <p>You can modify the field name and type for the analyzed schema.</p>
    </div>
  </div>
</template>

<script>
import { getSchemaColumnsData } from '@/api/workbench'

export default {
  name: 'Step1',
  data () {
    return {
      labelCol: { lg: { span: 5 }, sm: { span: 5 } },
      wrapperCol: { lg: { span: 19 }, sm: { span: 19 } },
      form: this.$form.createForm(this),
      schemaDataSource: [],
      schemaColumns: [
        {
          title: 'Column Name',
          dataIndex: 'name',
          scopedSlots: { customRender: 'name' }
        },
        {
          title: 'Column Type',
          dataIndex: 'type',
          scopedSlots: { customRender: 'type' }
        },
        {
          title: 'Comment',
          dataIndex: 'comment',
          scopedSlots: { customRender: 'comment' }
        }, {
          title: 'Operation',
          dataIndex: 'operation',
          scopedSlots: { customRender: 'operation' },
          width: 200
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
    },
    parseColumns () {
      getSchemaColumnsData().then(res => {
        this.schemaDataSource = res.result
        this.cacheData = this.schemaDataSource.map(item => ({ ...item }))
      })
    },
    handleAddColumn () {
      const { schemaDataSource } = this
      const newData = {
        key: 11,
        name: 'col_11',
        type: 'string',
        comment: 'comment ...',
        editable: false
      }
      this.schemaDataSource = [...schemaDataSource, newData]
    },

    onChangeColumn (value, key, column) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        target[column] = value
        this.schemaDataSource = newData
      }
    },

    onEditColumn (key) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        target.editable = true
        this.schemaDataSource = newData
      }
    },
    onSaveColumn (key) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        delete target.editable
        this.schemaDataSource = newData
        this.cacheData = newData.map(item => ({ ...item }))
      }
    },
    onCancelColumn (key) {
      const newData = [...this.schemaDataSource]
      const target = newData.filter(item => key === item.key)[0]
      if (target) {
        Object.assign(target, this.cacheData.filter(item => key === item.key)[0])
        delete target.editable
        this.schemaDataSource = newData
      }
    },

    handleDelColumn (key) {
      const schemaDataSource = [...this.schemaDataSource]
      this.schemaDataSource = schemaDataSource.filter(item => item.key !== key)
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
