<template>
  <div>
    <a-form :form="form" style="max-width: 650px; margin: 40px auto 0;">
      <a-form-item
        label="Data source"
        :labelCol="labelCol"
        :wrapperCol="wrapperCol"
      >
        <a-radio-group style="margin-right: 4px;" :value="dataSourceType" @change="handleDataSourceChange">
          <a-radio-button value="file">Upload File</a-radio-button>
          <a-radio-button value="hdfs">HDFS</a-radio-button>
          <a-radio-button value="s3" disabled>S3</a-radio-button>
        </a-radio-group>
      </a-form-item>
      <a-form-item
        label="Path"
        :labelCol="labelCol"
        :wrapperCol="wrapperCol">
        <a-input
          v-if="dataSourceType === 'file'"
          addonBefore="file://"
          placeholder="Please input path"/>
        <a-input
          v-if="dataSourceType === 'hdfs'"
          addonBefore="hdfs://"
          placeholder="Please input path"/>
      </a-form-item>
      <a-form-item
        v-if="dataSourceType === 'file'"
        label="Upload Files"
        :labelCol="labelCol"
        :wrapperCol="wrapperCol"
      >
        <a-upload
          :fileList="fileList"
          :remove="handleRemove"
          :beforeUpload="beforeUpload"
        >
          <a-button>
            <a-icon type="upload" /> Select File
          </a-button>
          <a-button
            type="primary"
            @click="handleUpload"
            :disabled="fileList.length === 0"
            :loading="uploading"
            style="margin-left: 8px;"
          >
            {{ uploading ? 'Uploading' : 'Start Upload' }}
          </a-button>
        </a-upload>
      </a-form-item>

      <a-form-item
        label="File Type"
        :labelCol="labelCol"
        :wrapperCol="wrapperCol"
      >
        <a-select placeholder="Please select file type">
          <a-select-option value="csv">csv</a-select-option>
          <a-select-option value="csv">csv</a-select-option>
          <a-select-option value="csv">csv</a-select-option>
        </a-select>
      </a-form-item>
      <a-form-item
        label="Column Delimiter"
        :labelCol="labelCol"
        :wrapperCol="wrapperCol"
      >
        <a-input style="width: 20%" v-decorator="['name', { initialValue: '.', rules: [{required: true, message: '收款人名称必须核对'}] }]"/>
        <a-checkbox style="padding-left: 10px;">First row is header</a-checkbox>
      </a-form-item>

      <a-form-item :wrapperCol="{span: 19, offset: 5}">
        <a-button type="primary" @click="nextStep" icon="profile">Create Table with UI</a-button>
        <a-button style="margin-left: 8px" icon="form">Create Table in Notebook</a-button>
      </a-form-item>
    </a-form>
    <a-divider />
    <div class="step-form-style-desc">
      <h3>Description</h3>
      <h4>Data source</h4>
      <p>Upload one or more local files to the store and create table structures and data based on the file to content.</p>
      <h4>Path</h4>
      <p>Specify the storage path of the file.</p>
      <h4>Upload Files</h4>
      <p>When using the `upload file` mode, add the local file to the upload list by clicking the `Select File` button. Click the `Start Upload` button to upload the file to the specified storage directory.</p>
      <h4>File Type</h4>
      <p>Select the type of file to upload, the system will parse the file according to the file type you choose.</p>
      <h4>Column Delimiter</h4>
      <p>Sets the separator for each column in the record, and the system splits the field based on the separator.</p>
      <h4>Create Table with UI</h4>
      <p>Use the UI operation interface to set the table schema, preview the data, and create the table step by step.</p>
      <h4>Create Table in Notebook</h4>
      <p>Create a table by handwriting the code through the notebook.</p>
    </div>
  </div>
</template>

<script>
import reqwest from 'reqwest'
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
    },
    handleRemove (file) {
      const index = this.fileList.indexOf(file)
      const newFileList = this.fileList.slice()
      newFileList.splice(index, 1)
      this.fileList = newFileList
    },
    beforeUpload (file) {
      this.fileList = [...this.fileList, file]
      return false
    },
    handleUpload () {
      const { fileList } = this
      const formData = new FormData()
      fileList.forEach((file) => {
        formData.append('files[]', file)
      })
      this.uploading = true

      // You can use any AJAX library you like
      reqwest({
        url: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
        method: 'post',
        processData: false,
        data: formData,
        success: () => {
          this.fileList = []
          this.uploading = false
          this.$message.success('upload successfully.')
        },
        error: () => {
          this.uploading = false
          this.$message.error('upload failed.')
        }
      })
    },
    handleDataSourceChange (e) {
      this.dataSourceType = e.target.value
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
