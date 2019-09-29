<template>
  <div>
    <s-table
      ref="table"
      size="default"
      :columns="tableColumns"
      :data="loadSampleData"
      bordered
    >
      <template slot="title">
        Sample Data
      </template>
    </s-table>
    <a-form :form="form" style="max-width: 650px; margin: 40px auto 0;">

      <a-form-item :wrapperCol="{span: 19, offset: 5}">
        <a-button @click="prevStep"><a-icon type="left" />Previous Step</a-button>
        <router-link :to="{ name: 'DataWrapper' }">
          <a-button style="margin-left: 8px" type="primary" @click="finish" icon="check">Submit</a-button>
        </router-link>
      </a-form-item>
    </a-form>
    <a-divider />
    <div class="step-form-style-desc">
      <h3>Description</h3>
      <h4>Sample Data Table</h4>
      <p>Convert the uploaded file into a table and display a part of the sample data.</p>
      <h4>Previous Step</h4>
      <p>If the sample data does not meet the requirements, you can go back to the previous steps and modify the configuration.</p>
      <h4>Submit</h4>
      <p>If the sample data meets the requirements, you can submit the table and data to the system and save it to the database.</p>
    </div>
  </div>
</template>

<script>
import { STable } from '@/components'
import { getSampleData, getTableColumns } from '@/api/workbench'

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
      // 表头
      tableColumns: [],

      // 加载数据方法 必须为 Promise 对象
      loadSampleData: parameter => {
        return getSampleData(Object.assign(parameter, this.queryParam))
          .then(res => {
            return res.result
          })
      }
    }
  },
  created () {
    getTableColumns().then(res => {
      this.tableColumns = res.result
      console.log('getTableColumns:' + this.tableColumns)
    })
  },
  methods: {
    finish () {
      this.$emit('finish')
    },
    prevStep () {
      this.$emit('prevStep')
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
